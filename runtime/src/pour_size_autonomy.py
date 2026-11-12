"""Live size-aware cup perception and the sealed headless pour batch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pour_guard_vision import cup_crop


SETTLED_RGB_KNOTS = (.755, .795, .835, .890)
MIDFILL_RGB_KNOTS = (.65, .70, .75, .82)
MAX_SETTLED_RGB_BIAS = .035


def settled_rgb_weight(value, knots=SETTLED_RGB_KNOTS):
    return float(np.interp(value, knots, (0., 1., 1., 0.)))


def sha(path):
    with Path(path).open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SizedCupPerception:
    def __init__(self, config, geometry, device="cuda"):
        from perception_followup import LevelFilter, VerifiedCupTracker
        self.device = device
        self.geometry = geometry
        self.tracker = VerifiedCupTracker(config["tracker"], geometry, device)
        self.cal = dict(config["tracker"]["calibration"], cup_geometry=geometry)
        self.models = config["models"]
        self.filters = {k: LevelFilter(v) for k, v in config["filters"].items()}
        self.agreement_limit = config["agreement_limit"]
        self.retry_initial_cup = config.get("retry_initial_cup", False)
        self.settled_rgb_calibration = config.get("settled_rgb_calibration")
        self.midfill_rgb_calibration = config.get("midfill_rgb_calibration")
        if self.settled_rgb_calibration and self.midfill_rgb_calibration:
            raise ValueError("Only one RGB calibration may be applied to a cup geometry")
        calibration = self.settled_rgb_calibration or self.midfill_rgb_calibration
        if calibration is not None:
            if (calibration["geometry"] != geometry
                    or not 0. <= calibration["bias"] <= MAX_SETTLED_RGB_BIAS):
                raise ValueError("RGB calibration does not match the declared cup or bias bounds")
        self.track_result = None

    def track(self, image):
        if self.retry_initial_cup and self.tracker.failed and self.tracker.anchor is None:
            # Restart consensus within the actor's original localization deadline.
            self.tracker.history.clear()
            self.tracker.failed = False
        if self.tracker.failed:
            self.track_result = dict(valid=False, reason="initial_reference_rejected", uv=None, score=0.)
        else:
            self.track_result = self.tracker.update(image)
        return self.track_result

    def level(self, image, xy, *, settling=False, draining=False):
        from perception_followup import predict_level
        raw, filtered = {}, {}
        track = self.track_result
        crop = torch.as_tensor(cup_crop(image, xy, self.cal), device=self.device)[None]
        for branch, model in self.models.items():
            value, visible = predict_level(crop, xy, self.geometry, model)
            valid = bool(visible[0]) and track["valid"] and bool(torch.isfinite(value[0]))
            raw[branch] = float(value[0]) if valid else None
        both_visible = all(v is not None for v in raw.values())
        for branch, value in raw.items():
            updated = self.filters[branch].update(value if both_visible else float("nan"))
            filtered[branch] = float(updated) if np.isfinite(updated) else None
        valid = all(v is not None for v in filtered.values())
        diagnostic = {}
        if valid and settling and self.settled_rgb_calibration is not None:
            correction = self.settled_rgb_calibration["bias"] * settled_rgb_weight(filtered["rgb"])
            diagnostic.update(guard_filtered=filtered.copy(), rgb_settle_correction=correction)
            filtered["rgb"] -= correction
        if valid and draining and self.midfill_rgb_calibration is not None:
            correction = self.midfill_rgb_calibration["bias"] * settled_rgb_weight(
                filtered["rgb"], MIDFILL_RGB_KNOTS)
            diagnostic.update(guard_filtered=filtered.copy(), rgb_midfill_correction=correction)
            filtered["rgb"] -= correction
        difference = abs(filtered["rgb"] - filtered["color"]) if valid else None
        return dict(track_valid=track["valid"], track_reason=track["reason"],
                    raw=raw, filtered=filtered, valid=valid,
                    disagreement=difference, agreement=valid and difference < self.agreement_limit,
                    **diagnostic)


def settled_pair(samples, target, tolerance, stability):
    """Require sustained, agreeing levels from both independently filtered branches."""
    if len(samples) < 6:
        return None, False, False
    pairs = np.asarray(samples[-6:], dtype=float)
    if not np.isfinite(pairs).all():
        return None, False, False
    low = float(np.quantile(pairs, .25, axis=0).min())
    stable = bool(np.std(pairs, axis=0).max() < stability)
    agree = bool(np.abs(pairs[:, 0] - pairs[:, 1]).max() < tolerance)
    return low, stable and agree, stable and agree and low >= target


def seeded_prefill(spec):
    from pour_perception_geometry import calibration_seed_points
    assert spec["policy_evaluation"] and spec["size_autonomy_prefill"]
    seed = Path(spec["calibration_particle_seed"])
    assert sha(seed) == spec["calibration_particle_seed_sha256"]
    with np.load(seed, allow_pickle=False) as archive:
        return calibration_seed_points(spec, archive["cup_local_points"])


def batch_specs(plan_path):
    """Check the sealed plan and select only the explicitly requested batch."""
    path = Path(plan_path)
    manifest = json.loads(path.read_text())
    assert manifest["run_authorized"]
    assert all(sha(n) == digest for n, digest in manifest["files"].items())
    names = [s["name"] for s in manifest["cases"]]
    assert names and len(set(names)) == len(names)
    for spec in manifest["cases"]:
        assert spec["policy_evaluation"] and spec["policy_family"] == "visual_guard_hierarchy"
        assert not any(spec.get(k) for k in ("calibration_only", "calibration_motion", "calibration_fill",
                                            "expert_recovery", "replay_episode", "impact_state_seed"))
        if spec.get("bottle_size_autonomy"):
            from pour_bottle_geometry import declared_geometry
            assert spec["declared_bottle_geometry"] == declared_geometry(spec)
    return manifest["cases"]
