"""Refit the full-view coarse cup heatmap for the shared pair-12 checkpoint."""
import inspect
import numpy as np
import torch

from diagnose_wide_mixed_r25 import PARENT, ROOT, STUDY
from diagnose_r22_startup import native_trials, setup
from fit_wide_mixed_r20 import state_for
from fit_wide_mixed_r7 import train_epoch
from fit_wide_mixed_r6 import save
from pour_cup_locator import CupLocator
from pour_guard_vision import project
from pour_cup_outline import OutlineCupTracker
from perception_followup import resize_coarse
from replay_r16_startup import erased, measure
from pour_size_autonomy import sha

KEY = "12_normal"
SEED = 26112412


def fit(cfg, *, revisions=(23,24), pair=12, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    images, labels, present, sources = [], [], [], []
    for revision in revisions:
        for case, frame in native_trials(ROOT / f"wide-mixed-r{revision}", {pair}):
            _, _, cal, z, xy = setup(case)
            uv = project([*xy,z],cal)
            sources.append(case["name"])
            for i in range(0,41,3):
                im = frame(i)
                images.append(resize_coarse(im))
                labels.append(uv*384/1024)
                present.append(True)
                if i % 2 == 0:
                    images.append(resize_coarse(erased(im,uv,"flat" if i%4 else "background")))
                    labels.append(labels[-1])
                    present.append(False)
    x = torch.as_tensor(np.stack(images), device="cuda")
    y = torch.as_tensor(np.stack(labels),device="cuda",dtype=torch.float32)
    has = torch.as_tensor(present,device="cuda")
    model = CupLocator().cuda()
    model.load_state_dict(cfg["tracker"]["coarse_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    yy,xx = torch.meshgrid(torch.arange(96,device="cuda"),torch.arange(96,device="cuda"),indexing="ij")
    for _ in range(24):
        train_epoch(model,optimizer,x,y,has,rng,xx,yy)
    weights = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    return weights, dict(seed=seed,epochs=24,learning_rate=2e-4,positive=sum(present),negative=present.count(False),
                         sources=sources)


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    path,state = state_for(KEY, PARENT)
    cfg = state["sized_cup_perception"]
    weights,training = fit(cfg)
    artifact = STUDY/"cup-coarse-12.pt"
    torch.save(weights,artifact)
    candidate = dict(cfg,tracker=dict(cfg["tracker"],coarse_state=weights))
    rows = []
    assert str(STUDY/"src") in inspect.getfile(OutlineCupTracker)
    for revision in (11,12,20,21,22,23,24):
        for case, frame in native_trials(ROOT / f"wide-mixed-r{revision}",{12}):
            result = measure(case,frame,OutlineCupTracker,STUDY,("12_normal","12_swap"),41,
                             parent_config=cfg,candidate_config=candidate)
            rows.append(result)
    positives = [r[k] for r in rows for k in ("candidate","translated_left","translated_right")]
    negatives = [r[k] for r in rows for k in ("flat","background")]
    target = next(r for r in rows if r["name"].endswith("r24_test_12_normal"))
    passed = len(rows)==14 and target["parent"] is None and all(v and v["error_mm"]<=2.5 for v in positives) and not any(negatives)
    result = dict(passed=bool(passed),checkpoint=path,artifact_sha256=sha(artifact),training=training,cases=rows,
                  positives=len(positives),negatives=len(negatives),false_acceptances=sum(v is not None for v in negatives))
    save(STUDY/"cup-startup-replay.json",result)
    print("cup",passed,[(r["name"],r["candidate"]) for r in rows],flush=True)
    assert passed


if __name__ == "__main__":
    main()
