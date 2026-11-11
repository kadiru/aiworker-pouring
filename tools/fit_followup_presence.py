"""Fit a cup-presence classifier with moving distractors and erased-cup challenges."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit_dynamic_rim import ROOT, protected_unchanged, read, sha
from perception_midfill_data import entries
from perception_followup import CupPresence, erased_coarse, presence_crop, resize_coarse
from pour_cup_locator import CupLocator, centers
from pour_guard_vision import project
from validate_pour_recording import decode, h5py


def prior_scenes():
    audit = ROOT / "holdout-r0-audit.json"
    review = read(ROOT / "holdout-r0-manual-review.json")
    result = read(ROOT / "moving-holdout-evaluation.json")
    assert review["accepted"] and review["audit_sha256"] == result["audit_sha256"] == sha(audit)
    specs = {s["name"]: s for s in read(ROOT / "protocol.json")["holdout"]}
    records = read(audit)["records"]
    assert len(records) == 18 and specs.keys() == {r["name"] for r in records}
    return [(specs[r["name"]], r, "regression") for r in records]


def decode_detail(data, group, index):
    return decode(data[group + "/cup_detail_image"][index], "jpeg95")


def source_path(spec, record):
    path = Path(record["path"])
    assert sha(path) == record["sha256"]
    actual = read(path.parent / "parameters.json")
    assert all(actual[k] == v for k, v in spec.items())
    return path


@torch.inference_mode()
def cache_images(loc, model):
    rows = {s: [[], []] for s in ("train", "validation")}
    cal = loc["calibration"]
    sources = []
    for spec, record, split in entries() + prior_scenes():
        path = source_path(spec, record)
        original_split = split
        if split == "regression":
            split = "train"
        world = np.asarray(spec["cup_world"], dtype=float)
        world[2] = cal["table_z"] + spec["declared_cup_geometry"]["rim_z_m"]
        truth = project(world, cal)
        with h5py.File(path) as f:
            data = f["data/demo_0"]
            indices = [("obs", i) for i in (0, 4, 7, 11)]
            times = data["next_time"][:]
            if spec.get("calibration_motion"):
                indices.extend(("next_obs", int(i)) for i in np.flatnonzero((times >= .5) & (times <= 2.5))[::2])
            else:
                indices.append(("next_obs", len(times) - 1))
            images = [decode_detail(data, group, i) for group, i in indices]
        sources.append(dict(name=spec["name"], split=original_split, sha256=record["sha256"], frames=len(images)))
        for start in range(0, len(images), 16):
            batch = images[start:start+16]
            small = [resize_coarse(im) for im in batch]
            erased = [erased_coarse(im, truth) for im in batch]
            tones = [erased_coarse(im, truth, 130 + (start+i) % 91) for i, im in enumerate(batch)]
            positive_logits = model(torch.as_tensor(np.stack(small), device="cuda"))
            positive, confidence = centers(positive_logits)
            absent, _ = centers(model(torch.as_tensor(np.stack(erased + tones), device="cuda")))
            yy, xx = torch.meshgrid(torch.arange(96, device="cuda")*4, torch.arange(96, device="cuda")*4, indexing="ij")
            outside = (xx-truth[0]*.375).square() + (yy-truth[1]*.375).square() > 68**2
            positive_logits[:, 0].masked_fill_(~outside, -100)
            distractors, _ = centers(positive_logits)
            for i, image in enumerate(batch):
                jitter = np.array([((start+i)*11)%21-10, ((start+i)*7)%17-8])
                pos = truth + jitter
                coarse = positive[i].cpu().numpy() / .375
                if np.linalg.norm(coarse-truth) < 34 and confidence[i] > loc["threshold"]:
                    pos = coarse
                rows[split][1].extend((presence_crop(image, pos), presence_crop(small[i], pos)))
                for j, masked in ((i, erased[i]), (i+len(batch), tones[i])):
                    rows[split][0].append(presence_crop(masked, absent[j].cpu().numpy() / .375))
                rows[split][0].append(presence_crop(erased[i], truth))
                rows[split][0].append(presence_crop(small[i], distractors[i].cpu().numpy() / .375))
    arrays = {f"{s}_{label}": np.stack(images) for s, groups in rows.items() for label, images in enumerate(groups)}
    return arrays, sources


@torch.inference_mode()
def confidence_range(model, data):
    score = torch.cat([model(batch).sigmoid() for batch in data.split(256)])
    return float(score.min()), float(score.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=36)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    torch.manual_seed(2611114070)
    protected_unchanged()
    fresh = read(ROOT / "followup-protocol.json")
    assert fresh["previous_protocol_sha256"] == sha(ROOT / "protocol.json")
    reserved = {s["name"] for s in fresh["holdout"]}
    assert not list(Path("/workspace/outputs/fountain").glob("*/*SD_A_motion_holdout_r1_*/parameters.json"))
    loc = torch.load(ROOT / "tracker-t3.pt", map_location="cpu", weights_only=False)
    cache = ROOT / "followup-presence-data.npz"
    metadata = ROOT / "followup-presence-data.json"
    if cache.exists():
        receipt = read(metadata)
        assert receipt["cache_sha256"] == sha(cache) and receipt["helper_sha256"] == sha(Path(__file__))
        with np.load(cache) as f:
            arrays = dict(f)
    else:
        coarse = CupLocator().cuda().eval()
        coarse.load_state_dict(loc["coarse_state"])
        arrays, sources = cache_images(loc, coarse)
        assert not reserved.intersection(s["name"] for s in sources)
        np.savez_compressed(cache, **arrays)
        receipt = dict(sources=sources, counts={k: len(v) for k, v in arrays.items()},
                       cache_sha256=sha(cache), helper_sha256=sha(Path(__file__)),
                       protocol_sha256=sha(ROOT / "followup-protocol.json"))
        metadata.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Cached", receipt["counts"], flush=True)
    data = {key: torch.as_tensor(value, device="cuda") for key, value in arrays.items()}
    del arrays
    model = CupPresence().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        pos, neg = data["train_1"], data["train_0"]
        batches = zip(torch.randperm(len(pos), device="cuda").split(64),
                      torch.randperm(len(neg), device="cuda").split(128))
        for pi, ni in batches:
            images = torch.cat((pos[pi], neg[ni])).float()
            gain = .92 + .16*torch.rand(len(images), 1, 1, 1, device="cuda")
            p, n = model(images*gain).split((len(pi), len(ni)))
            loss = F.softplus(-p).mean() + F.softplus(n).mean()
            loss += .25*(F.softplus(-p).topk(min(8, len(p))).values.mean()
                         + F.softplus(n).topk(min(8, len(n))).values.mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 3 and epoch != args.epochs:
            continue
        model.eval()
        bounds = {key: confidence_range(model, value) for key, value in data.items()}
        margin = min(bounds[s+"_1"][0] for s in ("train","validation")) - max(bounds[s+"_0"][1] for s in ("train","validation"))
        row = dict(epoch=epoch, bounds=bounds, margin=margin)
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or margin > best[0]["margin"]:
            best = row, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    selected, state = best
    loc.update(presence_state=state, accepted=False)
    output = ROOT / "followup-tracker.pt"
    assert not output.exists()
    torch.save(loc, output)
    protected_unchanged()
    report = dict(selected=selected, history=history, checkpoint_sha256=sha(output),
                  data_sha256=sha(metadata), protocol_sha256=sha(ROOT / "followup-protocol.json"),
                  helper_sha256=sha(Path(__file__)), parameters=sum(p.numel() for p in model.parameters()))
    with (ROOT / "followup-presence-fit.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
