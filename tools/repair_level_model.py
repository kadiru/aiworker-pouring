"""Study-only crop-tolerant RGB and independent color/outline level regressors.

Inputs are recorded RGB, image-derived cup position and declared dimensions.
Neither branch consumes simulator state or the other branch's predictions.
"""
import torch
from torch import nn
from scale_level_cnn import level_context
from scale_color_mask import mask_image


def color_outline(image):
    mask, valid = mask_image(image)
    a = image.to(torch.int64)
    gray = (77*a[..., 0] + 150*a[..., 1] + 29*a[..., 2]) // 256
    edge = torch.zeros_like(gray)
    edge[:, 1:-1, 1:-1] = (
        (gray[:, 2:, 1:-1] - gray[:, :-2, 1:-1]).abs()
        + (gray[:, 1:-1, 2:] - gray[:, 1:-1, :-2]).abs()).clamp(0, 255)
    red = (mask[..., 0] == 255).to(torch.int64)*255
    return torch.stack((red, gray, edge), -1).to(image.dtype), valid


class OutlineLevelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(1029, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, image, context):
        x = image.float().permute(0, 3, 1, 2)/255 - .5
        return self.head(torch.cat((self.encoder(x), context), 1))[:, 0].sigmoid()


_CACHE = {}


@torch.inference_mode()
def predict_repair(image, xy, geometry, artifact):
    key = (artifact['identity'], str(image.device))
    if key not in _CACHE:
        model = OutlineLevelNet().to(image.device).eval()
        model.load_state_dict(artifact['state'])
        _CACHE[key] = model
    x = image
    valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
    if artifact['branch'] == 'color':
        x, valid = color_outline(x)
    value = _CACHE[key](x, level_context(x, geometry, xy))
    return torch.where(valid, value, torch.full_like(value, float('nan'))), valid
