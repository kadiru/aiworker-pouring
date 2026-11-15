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


def validate_rgb_calibration(calibration, geometry, *, settling=False):
    knots = np.asarray(calibration["knots"])
    bias = calibration["bias"]
    if (calibration["geometry"] != geometry or knots.shape != (4,)
            or not np.isfinite(knots).all()
            or not 0. <= knots[0] < knots[-1] <= (.95 if settling else .90)
            or not np.all(np.diff(knots) > 0.) or not 0. <= bias <= MAX_SETTLED_RGB_BIAS
            or bias >= knots[1] - knots[0]):
        raise ValueError("Invalid size-specific RGB height calibration")


def calibrated_rgb_height(value, calibration):
    return value - calibration["bias"] * settled_rgb_weight(value, calibration["knots"])


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
        if config.get("outline_cup_localization"):
            from pour_cup_outline import OutlineCupTracker
            self.tracker = OutlineCupTracker(config["tracker"], geometry, device)
        else:
            self.tracker = VerifiedCupTracker(config["tracker"], geometry, device)
        self.cal = dict(config["tracker"]["calibration"], cup_geometry=geometry)
        self.models = config["models"]
        self.filters = {k: LevelFilter(v) for k, v in config["filters"].items()}
        self.agreement_limit = config["agreement_limit"]
        self.retry_initial_cup = config.get("retry_initial_cup", False)
        self.settled_rgb_calibration = config.get("settled_rgb_calibration")
        self.midfill_rgb_calibration = config.get("midfill_rgb_calibration")
        self.rgb_height_calibration = config.get("rgb_height_calibration", {})
        if self.rgb_height_calibration.keys() - {"settling", "draining"}:
            raise ValueError("Unsupported RGB height calibration phase")
        for phase, calibration in self.rgb_height_calibration.items():
            validate_rgb_calibration(calibration, geometry, settling=phase=="settling")
        for calibration in (self.settled_rgb_calibration, self.midfill_rgb_calibration):
            if calibration is None:
                continue
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
        phase = "settling" if settling else "draining" if draining else None
        calibration = self.rgb_height_calibration.get(phase)
        if valid and calibration is not None:
            diagnostic["guard_filtered"] = filtered.copy()
            diagnostic["calibrated_brake_level"] = max(
                calibrated_rgb_height(max(filtered["rgb"], raw["rgb"]), calibration),
                filtered["color"], raw["color"])
            filtered["rgb"] = calibrated_rgb_height(filtered["rgb"], calibration)
            diagnostic["rgb_height_phase"] = phase
        elif valid and settling and self.settled_rgb_calibration is not None:
            correction = self.settled_rgb_calibration["bias"] * settled_rgb_weight(filtered["rgb"])
            diagnostic.update(guard_filtered=filtered.copy(), rgb_settle_correction=correction)
            filtered["rgb"] -= correction
        elif valid and draining and self.midfill_rgb_calibration is not None:
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


def pulse_limit_level(samples, tolerance, stability):
    """Return the lower height across three consecutive settled windows."""
    if len(samples) < 8:
        return None
    recent = samples[-8:]
    results = [settled_pair(recent[i:i+6], 1., tolerance, stability) for i in range(3)]
    if all(stable for _,stable,_ in results):
        return min(level for level,_,_ in results)
    return None


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
