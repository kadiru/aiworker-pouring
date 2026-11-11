"""Re-crop recorded model inputs using camera-consistent XY perturbations.

No scene truth enters augmentation. Input positions are the learned tracker's
estimates. Camera calibration and declared cup dimensions are known inputs.
"""
import torch
from scale_rim_augmentation import affine


def crop_maps(xy, geometry, calibration):
    """Return the same destination-to-camera affine map as cup_crop."""
    xy = torch.as_tensor(xy)
    T = xy.new_tensor(calibration['T']); K = xy.new_tensor(calibration['K'])
    dims = xy.new_tensor([[g['rim_z_m'], g['bottom_z_m'], g['rim_radius_m']] for g in geometry])
    center = torch.cat((xy, xy.new_full((len(xy), 1), calibration['table_z'])), 1)
    rim = center.clone(); rim[:, 2] += dims[:, 0]
    base = center.clone(); base[:, 2] += dims[:, 1]

    def project(world):
        v = (world-T[:3, 3])@T[:3, :3]
        return torch.stack((K[0, 0]*v[:, 0]/(-v[:, 2])+K[0, 2],
                            K[1, 2]-K[1, 1]*v[:, 1]/(-v[:, 2])), 1)

    r = project(rim); b = project(base)
    radius = (project(rim+T[:3, 0]*dims[:, 2:3])[:, 0]-r[:, 0]).abs()
    sx = radius/32; shear = (b[:, 0]-r[:, 0])/40; sy = (b[:, 1]-r[:, 1])/40
    zero = torch.zeros_like(sx)
    matrix = torch.stack((sx, shear, zero, sy), 1).reshape(-1, 2, 2)
    offset = torch.stack((r[:, 0]-48*sx-24*shear, r[:, 1]-24*sy), 1)
    return matrix, offset


def perturb_crop(image, xy, geometry, calibration, delta_xy):
    old, old_offset = crop_maps(xy, geometry, calibration)
    new, new_offset = crop_maps(xy+delta_xy, geometry, calibration)
    # A point p in the old crop moves to q in the perturbed crop.
    inverse = torch.linalg.inv(new)
    matrix = inverse@old
    translation = (inverse@(old_offset-new_offset)[..., None])[..., 0]
    center = image.new_tensor([(image.shape[2]-1)/2, (image.shape[1]-1)/2])
    shift = (matrix@center[None, :, None])[..., 0]+translation-center
    transformed, _ = affine(image, image.new_zeros(len(image), 2), matrix, shift)
    return transformed.clamp(0, 255)


def random_offsets(xy, radius_m):
    angle = torch.rand(len(xy), device=xy.device)*2*torch.pi
    radius = torch.rand(len(xy), device=xy.device).sqrt()*radius_m
    return torch.stack((angle.cos()*radius, angle.sin()*radius), 1)
