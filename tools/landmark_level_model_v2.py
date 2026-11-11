"""Geometry-conditioned relative landmarks with full-precision convolutions."""
import torch
from torch import nn
from landmark_level_model import LandmarkLevelNet as BaseLandmarkNet, points_from_logits, level_from_points
from repair_level_model import color_outline


class LandmarkLevelNet(BaseLandmarkNet):
    def __init__(self):
        super().__init__()
        self.net[0] = nn.Conv2d(6, 16, 5, 2, 2)

    def forward(self, image, geometry):
        if not isinstance(geometry, list): geometry = [geometry]*len(image)
        dims = image.new_tensor([[(g['rim_z_m']-g['bottom_z_m'])/.034,
                                  g['rim_radius_m']/.024] for g in geometry], dtype=torch.float32)
        context = (torch.cat((dims, dims[:, 1:2]/dims[:, :1]), 1)-1)*5
        rgb = image.float().permute(0, 3, 1, 2)/255.
        x = torch.cat((rgb, context[:, :, None, None].expand(-1, -1, image.shape[1], image.shape[2])), 1)
        # Ray-based height magnifies tiny landmark differences; keep CUDA/CPU
        # arithmetic paths close without changing other models' global settings.
        with torch.backends.cudnn.flags(allow_tf32=False):
            return self.net(x)


def initialize_from_base(model, state):
    expanded = model.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            if name == 'net.0.weight':
                expanded[name].zero_(); expanded[name][:, :3].copy_(value)
            else: expanded[name].copy_(value)
    model.load_state_dict(expanded)


_CACHE = {}


@torch.inference_mode()
def predict_landmark_level(image, xy, geometry, artifact):
    key = (artifact['identity'], str(image.device))
    if key not in _CACHE:
        model = LandmarkLevelNet().to(image.device).eval()
        model.load_state_dict(artifact['state']); _CACHE[key] = model
    x = image; valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
    if artifact['branch'] == 'color': x, valid = color_outline(x)
    points = points_from_logits(_CACHE[key](x, geometry))
    value = level_from_points(points, xy, geometry, artifact['calibration'])
    valid = valid & torch.isfinite(value)
    return torch.where(valid, value.clamp(0, 1), torch.full_like(value, float('nan'))), valid
