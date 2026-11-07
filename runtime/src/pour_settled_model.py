"""Image-derived features for a supervised final-settled-height regressor."""
import numpy as np
import torch
from pour_guard_vision import surface_features


def features(image, cnn, cup_xy):
    f, hist, mass = surface_features(image)
    xy = torch.as_tensor(cup_xy, device=image.device, dtype=f.dtype)
    if xy.ndim == 1:
        xy = xy[None].expand(len(image), -1)
    base = torch.stack((f[:, 0], f[:, 1], f[:, 2], f[:, 3], cnn,
        (xy[:, 0] - .49) / .008, (xy[:, 1] + .15) / .008), 1)
    # Keep the vertical color distribution: a mean alone loses rim-level detail.
    bands = hist[:, 14:84].reshape(len(image), 14, 5).mean(2)
    original = torch.cat((base, bands), 1)
    # Perspective changes how each visual level feature maps to rim height.
    # Preserve the original 21 columns for existing checkpoints, then expose
    # spatial interactions for the wider-table supervised candidate.
    view = base[:, 5:7] * .08
    visual = torch.cat((base[:, 1:5], bands), 1)
    legacy = torch.cat((original, visual * view[:, :1], visual * view[:, 1:],
                        view[:, :1].square(), view[:, :1] * view[:, 1:],
                        view[:, 1:].square()), 1)
    # Five-row pooling loses the small surface displacement that matters near
    # the rim. Preserve row detail and continuous red evidence at color edges.
    rgb = image.float() / 255.
    r, g, b = rgb.unbind(-1)
    red = torch.sigmoid((r - 1.4*g - .025) / .04) * torch.sigmoid((r - 1.2*b - .02) / .04)
    fine = hist[:, 14:84]
    soft = red[:, 14:84, 35:73].mean(2)
    existing = torch.cat((legacy, fine, soft, fine * view[:, :1], fine * view[:, 1:]), 1)
    # Color-envelope quantiles retain a compact level signal across cup views.
    cumulative = hist.cumsum(1) / hist.sum(1, keepdim=True).clamp(min=1e-6)
    qs = torch.tensor([.02,.05,.1,.25,.5,.75,.9,.95,.98],device=image.device)
    quantiles = (cumulative[:,:,None] >= qs).float().argmax(1).float() / 95.
    return torch.cat((existing, quantiles, quantiles*view[:,:1], quantiles*view[:,1:]),1)


def predict(x, fitted):
    columns = fitted['columns']
    a = (x[:, columns] - x.new_tensor(fitted['mean'])) / x.new_tensor(fitted['scale'])
    return (a @ x.new_tensor(fitted['coef'])).clamp(0., 1.)


def assess(levels, other, target, deviation, tolerance, mass):
    values = np.asarray(levels, dtype=float)
    if len(values) < 6 or not np.isfinite(values).all():
        return 0., False, False
    level = float(np.quantile(values, .25))
    stable = bool(np.std(values) < deviation)
    return level, stable, bool(level >= target and abs(other - level) < tolerance and stable and mass > 1.)


def topup_cap(previous_peak, extra=.25, hard_limit=110.):
    return min(hard_limit, max(95., previous_peak + extra))


def topup_feedback_eligible(phase, cap):
    # RGB changes as the bottle moves are not proof that a top-up has flowed.
    return phase >= cap - .25


def pulse_finished(phase, cap, elapsed, timeout):
    """End a short motion pulse; level feedback is evaluated only at rest."""
    return phase >= cap - 1e-6 or elapsed >= timeout
