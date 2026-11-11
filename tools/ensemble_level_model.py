"""Fixed RGB ensemble and independent color model, with full-precision inference."""
import torch
from repair_level_model_v2 import predict_repair
from landmark_level_model import predict_landmark_level


@torch.inference_mode()
def predict_ensemble_level(image, xy, geometry, artifact):
    kind = artifact['kind']
    if kind == 'rgb_equal_ensemble_v1':
        direct, landmarks = artifact['members']
        if (artifact['branch'] != 'rgb' or artifact['weights'] != [.5, .5]
                or direct['branch'] != 'rgb' or landmarks['branch'] != 'rgb'
                or direct['kind'] != 'crop_tolerant_outline_v2'
                or landmarks['kind'] != 'geometry_landmarks_v1'):
            raise ValueError('RGB ensemble requires the frozen RGB-only model families and equal weights')
    elif kind == 'full_precision_color_landmarks_v1':
        landmarks = artifact['member']
        if artifact['branch'] != 'color' or landmarks['branch'] != 'color' or landmarks['kind'] != 'geometry_landmarks_v1':
            raise ValueError('Color cross-check requires its independent color weights')
    else:
        raise ValueError('Unknown ensemble artifact')
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(allow_tf32=False):
            b, valid = predict_landmark_level(image, xy, geometry, landmarks)
            value = b
            if kind == 'rgb_equal_ensemble_v1':
                a, av = predict_repair(image, xy, geometry, direct)
                valid = valid & av & torch.isfinite(a)
                value = .5*a + .5*b
            valid = valid & torch.isfinite(value)
            return torch.where(valid, value, torch.full_like(value, float('nan'))), valid
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
