"""Reconcile the four particle inventories at every recorded test sample.

Run on the Mac after the final native manifest and backups have been verified.
This is an offline observation audit; it does not change the benchmark score.
"""
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence/workspace-A"
DATA = PROJECT / "data/material-A-workspace"
NATIVE = Path("/workspace/outputs/fountain")
COMPARTMENTS = ("bottle_ml", "cup_ml", "airborne_ml", "spilled_ml")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_measurement(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            same_measurement(v, b[k]) for k, v in a.items())
    # Local and native math.hypot can differ by one ulp. Particle counts,
    # artifact identities, labels and other non-floating values remain exact.
    if isinstance(a, float):
        return isinstance(b, float) and math.isclose(a, b, rel_tol=0., abs_tol=1e-9)
    return type(a) is type(b) and a == b


def particles(nominal_ml, quantum):
    count = round(nominal_ml / quantum)
    assert count >= 0 and math.isclose(nominal_ml, count * quantum, rel_tol=0, abs_tol=1e-9)
    return count


def mouth_geometry(sample, spec):
    # Rotate the local bottle-mouth center (0, 0, 135 mm) into world space.
    # The receiver is fixed during these tests. Radii here are geometry, not
    # a stream footprint: being over the opening does not guarantee a catch.
    x, y, z, w = sample["bottle_quat_xyzw"]
    assert math.isclose(x*x + y*y + z*z + w*w, 1, abs_tol=1e-5)
    axis = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
    dx, dy, dz = [1000*(a-b) + 135*u for a, b, u in
                  zip(sample["bottle_pose_xyz"], spec["cup_world"], axis)]
    return dict(t=sample["t"], radial_offset_mm=math.hypot(dx, dy),
                height_above_rim_mm=dz-38, tilt_deg=sample["bottle_tilt_deg"])


def check_record(folder, row):
    metrics = read(folder / "metrics.json")
    spec = read(folder / "parameters.json")
    samples = read(folder / "samples.json")
    quantum = metrics["spill"]["particle_quantum_ml"]
    assert spec["name"] == row["name"] and math.isclose(quantum, spec["spacing"] ** 3 * 1e6)
    assert samples
    total = samples[0]["count"]
    previous_t, ever, first_near, peak_outside, outside_samples = -1, 0, None, 0, 0
    for s in samples:
        counts = [particles(s[k], quantum) for k in COMPARTMENTS]
        assert s["finite"] and s["count"] == total == sum(counts), (spec["name"], s["t"], counts)
        assert s["t"] > previous_t
        previous_t = s["t"]
        near = particles(s["ever_table_ml"], quantum)
        assert ever <= near <= total
        if near and first_near is None:
            first_near = s["t"]
        ever = near
        peak_outside = max(peak_outside, counts[3])
        outside_samples += counts[3] > 0
    source, cup, airborne, outside = counts
    assert source == particles(row["source_remaining_ml"], quantum)
    assert cup == particles(row["cup_final_ml"], quantum)
    assert airborne + outside == particles(row["final_outside_ml"], quantum)
    assert ever == particles(row["ever_below_worktop_ml"], quantum)
    assert row["zero_spill"] == (airborne + outside == 0 and ever == 0)
    # For exact-empty runs, compare the same physical event across layouts:
    # the first observation in the final stretch of zero remaining particles.
    # Residual-but-allowed or incomplete drains have no such event.
    terminal_empty = None
    if spec["expected_stop"] == "bottle_empty" and source == 0:
        drain_i = next((i + 1 for i in range(len(samples)-1, -1, -1)
                        if samples[i]["bottle_ml"] > 0), 0)
        terminal_empty = mouth_geometry(samples[drain_i], spec)
        terminal_empty["airborne_outside_particles"] = particles(samples[drain_i]["airborne_ml"], quantum)
        terminal_empty["nonairborne_outside_particles"] = particles(samples[drain_i]["spilled_ml"], quantum)
    loss_timing = None
    if ever or airborne + outside:
        decisions = read(folder / "policy-decisions.json")
        stop = next((d for d in decisions if d["stop_latched"]), None)
        first = next((s for s in samples if s["ever_table_ml"] > 0), None)
        def snapshot(s):
            if s is None:
                return None
            return dict(**mouth_geometry(s, spec), phase=s["phase"],
                        source_particles=particles(s["bottle_ml"], quantum),
                        cup_particles=particles(s["cup_ml"], quantum),
                        airborne_outside=particles(s["airborne_ml"], quantum),
                        nonairborne_outside=particles(s["spilled_ml"], quantum),
                        ever_near_table=particles(s["ever_table_ml"], quantum))
        at = min(samples, key=lambda s: abs(s["t"]-stop["t"])) if stop else None
        if at is not None:
            assert abs(at["t"]-stop["t"]) < .051
        loss_timing = dict(stop_s=stop["t"] if stop else None,
                           stop_reason=stop["stop_latched"] if stop else None,
                           at_stop=snapshot(at), first_near_table=snapshot(first),
                           first_near_table_minus_stop_s=first["t"]-stop["t"] if first and stop else None)
    return dict(name=spec["name"], samples=len(samples), total_particles=total,
                final_source=source, final_cup=cup, final_airborne=airborne,
                final_nonairborne_outside=outside, ever_near_table=ever,
                first_near_table_s=first_near, peak_nonairborne_outside=peak_outside,
                nonairborne_outside_samples=outside_samples, nominal_ml_per_particle=quantum,
                last_drain_geometry=terminal_empty, loss_timing=loss_timing)


def main():
    report = read(EVIDENCE / "final-benchmark.json")
    assert report["independently_recomputed_primary_scores"] == len(report["cases"]) == 30
    integrity = read(EVIDENCE / "recording-integrity.json")
    verification = read(EVIDENCE / "final-backup-verification.json")
    assert verification["all_native_hashes_and_sizes_match"]
    assert verification["integrity_manifest_sha256"] == digest(EVIDENCE / "recording-integrity.json")
    indexed = {f["relative_path"]: f for group in ("files", "sidecar_files") for f in integrity[group]}
    # Preserve the attribution of early spill evidence, intermediate backups, and prespecified videos.
    archive_names = ["first-spill-backup-manifest.json"] + [
        f"backup-test-block-{span}.json" for span in ("00-04", "05-14", "15-24")]
    for name in archive_names:
        archive = read(EVIDENCE / name)
        for group in ("files", "sidecar_files"):
            for f in archive[group]:
                final = indexed[f["relative_path"]]
                assert (f["sha256"], f["bytes"]) == (final["sha256"], final["bytes"]), name
    for label in ("full", "empty"):
        v = read(EVIDENCE / f"fresh-{label}-video-review.json")
        source = indexed[str(Path(v["source_hdf5"]).relative_to(NATIVE))]
        assert v["source_sha256"] == source["sha256"] and v["name"] == source["name"]
        assert v["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert digest(EVIDENCE / v["video"]) == v["video_sha256"]
        assert digest(EVIDENCE / v["review"]["sheet"]) == v["review"]["sheet_sha256"]
    rows = [check_record(DATA / Path(r["path"]).relative_to(NATIVE), r) for r in report["cases"]]
    by_name = {r["name"]: r for r in rows}
    folders = {r["name"]: Path(r["path"]).relative_to(NATIVE) for r in report["cases"]}
    earlier_losses = read(EVIDENCE / "first-two-losses-comparison.json")["rows"]
    rejection = read(EVIDENCE / "test16-localization-diagnostic.json")
    underfill = read(EVIDENCE / "test22-full-stop-diagnostic.json")
    missed_stop = read(EVIDENCE / "test24-missed-empty-diagnostic.json")
    for diagnostic in [rejection, underfill, missed_stop, *earlier_losses]:
        folder = folders[diagnostic["name"]]
        for name, native_hash in diagnostic["sidecar_sha256"].items():
            assert indexed[str(folder / name)]["sha256"] == native_hash
    for earlier in earlier_losses:
        assert same_measurement(by_name[earlier["name"]],
                                {k: v for k, v in earlier.items() if k != "sidecar_sha256"})
    summary = dict(checkpoint_sha256=report["checkpoint_sha256"], cases=len(rows),
                   samples=sum(r["samples"] for r in rows), all_sampled_totals_conserved=True,
                   verified_earlier_manifests=archive_names, verified_video_sources_and_exports=2,
                   verified_interim_failure_diagnostics=3 + len(earlier_losses),
                   zero_final_and_near_table=sum(r["ever_near_table"] == 0 and
                     r["final_airborne"] + r["final_nonairborne_outside"] == 0 for r in rows),
                   cases_with_nonairborne_outside=sum(r["nonairborne_outside_samples"] > 0 for r in rows),
                   max_nonairborne_outside_particles=max(r["peak_nonairborne_outside"] for r in rows),
                   note="Four compartments sum to the initial particle count at every saved sample. "
                        "This checks accounting at 20 Hz, not trajectories between observations.")
    (EVIDENCE / "particle-accounting.json").write_text(json.dumps(dict(**summary, rows=rows), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
