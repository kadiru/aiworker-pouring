"""Seal the final-droplet correction with the original unconsumed test layouts."""
import copy
from pathlib import Path
import shutil

import torch

from prepare_layout_repair import ROOT, STUDY as PREVIOUS, folder_for, read, save, sha

STUDY = ROOT / "layout-repair-r3"


def main():
    result_path = PREVIOUS / "development-results.json"
    previous_plan = PREVIOUS / "plan.json"
    plan, development = read(previous_plan), read(result_path)
    assert development["plan_sha256"] == sha(previous_plan)
    assert development["completed"] == development["planned"] == 4
    failures = [row for row in development["cases"] if not row["strict_pass"]]
    assert len(failures) == 1 and failures[0]["name"].endswith("04_empty")
    short = failures[0]
    assert short["failure_reasons"] == ["measured_spill"] and short["within_particle_tolerance"]
    assert short["source_at_stop_ml"] == short["source_remaining_ml"] == 0.
    assert short["empty_confirmation_method"] == "drain_dwell"
    assert not (PREVIOUS / "test-results.json").exists()
    fresh = {scene["name"] for scene in plan["test"]}
    assert not any(read(path)["name"] in fresh
                   for path in Path("/workspace/outputs/size-autonomy").glob("*/*/*SA_A_*/parameters.json"))

    folder = folder_for(short)
    samples_path = folder / "samples.json"
    samples = read(samples_path)
    first_loss = next(row for row in samples if row["spilled_ml"] > 0.)
    final_outside = [row["index"] for row in samples[-1]["diagnostic_outside_particles"]]
    traces = []
    for index in final_outside:
        early = next((row, particle) for row in samples
                     for particle in row["diagnostic_outside_particles"] if particle["index"] == index)
        traces.append(dict(time_s=early[0]["t"], tilt_deg=early[0]["bottle_tilt_deg"],
                           xyz_cup_frame_m=early[1]["xyz_cup_frame_m"], velocity_m_s=early[1]["velocity_m_s"]))
    assert len(traces) == short["measured_loss_particles"] == 2
    assert all(p["xyz_cup_frame_m"][0] > 0 and 123. <= p["tilt_deg"] <= 131. for p in traces)

    template = next(s for s in plan["development"] if s["name"] == short["name"])
    source = template["checkpoint"]
    assert sha(source) == plan["checkpoints"][source]
    saved = torch.load(source, map_location="cpu", weights_only=False)
    assert saved["short_empty_requires_drain"] and saved["late_mouth_shift_m"] == .008
    saved.update(tail_setback_m=.008, model_version="visual_guard_layout_repair_r3")
    weight = STUDY / "short-narrow.pt"
    assert not weight.exists()
    torch.save(saved, weight)
    plan["checkpoints"][str(weight)] = sha(weight)
    assert sum(s["checkpoint"] == source for s in plan["test"]) == 1
    for scene in plan["test"]:
        if scene["checkpoint"] == source:
            scene["checkpoint"] = str(weight)
    del plan["checkpoints"][source]
    verify = copy.deepcopy(template)
    verify.update(name="SA_A_layout_r3_development_04_empty", seed=2611124104, checkpoint=str(weight))
    plan["development"] = [verify]
    plan["development_requires_strict_zero"] = False
    plan["controller_scope"] = "Verify the short bottle's final-drop alignment, then run the six reserved layouts"
    plan["prerequisite_reports"].update({str(result_path): sha(result_path), str(previous_plan): sha(previous_plan)})

    previous_replay = PREVIOUS / "empty-replay.json"
    assert sha(previous_replay) == plan["empty_confirmation_replay_sha256"]
    receipt = read(previous_replay)
    receipt["short_tail_alignment"] = dict(
        setback_m=.008, ramp_deg=[123., 126.], source_case=short["name"],
        source_recording_sha256=short["recording_sha256"], samples_sha256=sha(samples_path),
        first_loss_s=first_loss["t"], measured_loss_particles=short["measured_loss_particles"],
        late_droplet_traces=traces, prior_strict_passes=[r["name"] for r in development["cases"] if r["strict_pass"]],
        development_particle_limit=plan["provisional_particle_limit"])
    save(STUDY / "empty-replay.json", receipt)
    plan["empty_confirmation_replay_sha256"] = sha(STUDY / "empty-replay.json")
    shutil.copy2(PREVIOUS / "calibration.json", STUDY / "calibration.json")
    save(STUDY / "plan.json", plan)
    print(receipt["short_tail_alignment"], flush=True)


if __name__ == "__main__":
    main()
