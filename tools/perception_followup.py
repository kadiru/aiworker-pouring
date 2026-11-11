"""Presence verification and causal level filtering for the stationary cup study."""
from collections import deque

import numpy as np
from PIL import Image
import torch
from torch import nn

from dynamic_cup_tracker import CupTracker
from pour_cup_locator import centers
from scale_rim_refiner import crop_native
from ensemble_level_model import predict_ensemble_level


def resize_coarse(image):
    return np.asarray(Image.fromarray(image).resize((384, 384), Image.Resampling.BILINEAR)).copy()


def presence_crop(image, point):
    # Work in 1024-coordinate units, including for the original low-resolution challenge.
    scale = image.shape[0] / 1024
    crop, _ = crop_native(image, (np.asarray(point) + (0, 24)) * scale, round(224 * scale))
    return np.asarray(Image.fromarray(crop).resize((80, 80), Image.Resampling.BILINEAR)).copy()


def erased_coarse(image, point, tone=180):
    small = resize_coarse(image)
    x, y = np.rint(np.asarray(point) * 384 / 1024).astype(int)
    small[max(0, y-24):min(384, y+48), max(0, x-50):min(384, x+50)] = tone
    return small


class CupPresence(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(512, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, crops):
        return self.layers(crops.permute(0, 3, 1, 2).float() / 255).flatten()


class VerifiedCupTracker(CupTracker):
    def __init__(self, artifact, geometry, device="cuda", presence=None):
        super().__init__(artifact, geometry, device)
        self.presence = presence
        if self.presence is None:
            self.presence = CupPresence().to(device).eval()
            self.presence.load_state_dict(artifact["presence_state"])

    def presence_score(self, image, point):
        if not np.isfinite(point).all():
            return 0.
        crop = torch.as_tensor(presence_crop(image, point), device=self.device)[None]
        return float(self.presence(crop).sigmoid()[0])

    @torch.inference_mode()
    def detect(self, image, refine=True):
        point, score = super().detect(image, refine=refine)
        return point, min(score, self.presence_score(image, point))

    @torch.inference_mode()
    def absent_score(self, small):
        uv, score = centers(self.coarse(torch.as_tensor(small, device=self.device)[None]))
        point = uv[0].cpu().numpy() * (1024 / 384)
        return min(float(score[0]), self.presence_score(small, point))

    @torch.inference_mode()
    def update(self, image):
        if self.anchor is None:
            return super().update(image)
        # Confirm the lower outline in the current frame plus either learned
        # presence or a coarse detection that agrees with the fixed anchor.
        score = self.presence_score(image, self.anchor)
        reason = "tracked"
        settings = self.artifact["settings"]
        if score < self.artifact["threshold"]:
            point, coarse_score = super().detect(image, refine=False)
            if np.isfinite(point).all() and np.linalg.norm(point - self.anchor) <= settings["association_pixels"]:
                score = max(score, coarse_score)
                reason = "tracked_with_coarse_confirmation"
        if score < self.artifact["threshold"]:
            return dict(valid=False, reason="cup_absent_or_uncertain", uv=None, score=score)
        similarity, movement, shift = self.match(image)
        valid = similarity >= settings["minimum_similarity"] and movement <= settings["maximum_movement_mm"]
        return dict(valid=valid, reason=reason if valid else "cup_moved_or_reference_changed",
                    uv=self.anchor.tolist() if valid else None, score=score,
                    similarity=similarity, movement_mm=movement, shift_pixels=shift)


class LevelFilter:
    def __init__(self, config):
        self.values = deque(maxlen=config["window"])
        self.current_weight = config["current_weight"]

    def update(self, value):
        if not np.isfinite(value):
            self.values.clear()
            return float("nan")
        self.values.append(float(value))
        return self.current_weight * value + (1 - self.current_weight) * float(np.median(self.values))


def predict_level(image, xy, geometry, artifact):
    return predict_ensemble_level(image, xy, geometry, artifact)
