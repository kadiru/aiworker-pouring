"""Relative rim/surface landmarks with calibrated height reconstruction."""
import torch
from torch import nn
from physical_level_augmentation import crop_maps
from repair_level_model import color_outline


class LandmarkLevelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(32, 2, 1))

    def forward(self, image):
        return self.net(image.float().permute(0, 3, 1, 2)/255.)


def points_from_logits(logits):
    h, w = logits.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=logits.device, dtype=logits.dtype)*4+2,
                          torch.arange(w, device=logits.device, dtype=logits.dtype)*4+2, indexing='ij')
    probability = logits.flatten(2).softmax(2).reshape_as(logits)
    return torch.stack(((probability*x).sum((-2, -1)), (probability*y).sum((-2, -1))), -1)


def level_from_points(points, xy, geometry, calibration):
    """Rim image ray fixes XY at known rim Z; surface ray determines water Z.

    `xy` is only the crop's image-estimated origin. The inferred rim landmark
    corrects it geometrically; no true scene position is read at inference.
    """
    points = points.double()
    xy = torch.as_tensor(xy, device=points.device, dtype=torch.float64).reshape(-1, 2)
    if len(xy) == 1: xy = xy.expand(len(points), 2)
    if not isinstance(geometry, list): geometry = [geometry]*len(points)
    A, b = crop_maps(xy, geometry, calibration)
    uv = (A[:, None]@points[..., None])[..., 0]+b[:, None]
    T = points.new_tensor(calibration['T']); K = points.new_tensor(calibration['K'])
    camera_rays = torch.stack(((uv[..., 0]-K[0, 2])/K[0, 0],
                              -(uv[..., 1]-K[1, 2])/K[1, 1], -torch.ones_like(uv[..., 0])), -1)
    rays = camera_rays@T[:3, :3].T
    rim_z = points.new_tensor([calibration['table_z']+g['rim_z_m'] for g in geometry])
    bottom_z = points.new_tensor([calibration['table_z']+g['bottom_z_m'] for g in geometry])
    rim = T[:3, 3]+rays[:, 0]*((rim_z-T[2, 3])/rays[:, 0, 2])[:, None]
    horizontal = rays[:, 1, :2]
    distance = ((rim[:, :2]-T[:2, 3])*horizontal).sum(1)/horizontal.square().sum(1).clamp_min(1e-12)
    surface_z = T[2, 3]+distance*rays[:, 1, 2]
    return (surface_z-bottom_z)/(rim_z-bottom_z)


_CACHE = {}


@torch.inference_mode()
def predict_landmark_level(image, xy, geometry, artifact):
    key = (artifact['identity'], str(image.device))
    if key not in _CACHE:
        model = LandmarkLevelNet().to(image.device).eval()
        model.load_state_dict(artifact['state']); _CACHE[key] = model
    x = image; valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
    if artifact['branch'] == 'color': x, valid = color_outline(x)
    points = points_from_logits(_CACHE[key](x))
    value = level_from_points(points, xy, geometry, artifact['calibration'])
    valid = valid & torch.isfinite(value)
    return torch.where(valid, value.clamp(0, 1), torch.full_like(value, float('nan'))), valid
