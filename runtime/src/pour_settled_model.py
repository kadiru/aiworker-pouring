"""Image-derived features for a supervised final-settled-height regressor."""
import numpy as np
import torch
from pour_guard_vision import surface_features


def stable_color_quantiles(hist):
    """Nearest-rank red-row quantiles with exact integer count comparisons.

    surface_features averages a binary mask across columns 35:73 (38 pixels).
    Recover those integer counts so a cumulative sum exactly on a quantile
    boundary cannot choose different rows on CPU and GPU.
    """
    counts=(hist*38).round().to(torch.int64)
    cumulative=counts.cumsum(1);total=counts.sum(1,keepdim=True)
    percent=torch.tensor([2,5,10,25,50,75,90,95,98],device=hist.device,dtype=torch.int64)
    return (cumulative[:,:,None]*100>=total[:,:,None]*percent).to(torch.int64).argmax(1).to(hist.dtype)/95.


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
    quantiles = stable_color_quantiles(hist)
    quadratic = torch.cat((view[:,:1].square(),view[:,:1]*view[:,1:],view[:,1:].square()),1)
    envelope = torch.cat((quantiles,quantiles*view[:,:1],quantiles*view[:,1:]),1)
    perspective = torch.cat([torch.cat((quantiles,fine),1)*quadratic[:,i:i+1] for i in range(3)],1)
    legacy_all=torch.cat((existing,envelope,perspective),1)
    # Row averages discard the ellipse's lateral shape. Keep continuous red
    # evidence in a small 2-D grid and in thirteen vertical surface profiles.
    # A positive color margin excludes dark cup walls from the soft mask.
    strength=(torch.minimum(r-1.4*g-.025,r-1.2*b-.02)/.12).clamp(0,1)
    region=strength[:,12:84,30:82]
    pooled=torch.nn.functional.adaptive_avg_pool2d(region[:,None],(8,8)).flatten(1)
    profiles=region.reshape(len(image),72,13,4).mean(3)
    profile_mass=profiles.sum(1);normalized=profiles/profile_mass[:,None].clamp(min=1e-6)
    yy=torch.arange(12,84,device=image.device,dtype=normalized.dtype)[None,:,None]/95.
    center=(normalized*yy).sum(1)
    spread=(normalized*(yy-center[:,None]).square()).sum(1).sqrt()
    total=profile_mass.sum(1,keepdim=True)
    global_center=(center*profile_mass).sum(1,keepdim=True)/total.clamp(min=1e-6)
    shape=torch.cat((pooled,center,spread,profile_mass/72.,global_center,total/(72*13)),1)
    return torch.cat((legacy_all,shape),1)


def predict(x, fitted):
    columns = fitted['columns']
    a = (x[:, columns] - x.new_tensor(fitted['mean'])) / x.new_tensor(fitted['scale'])
    value = a @ x.new_tensor(fitted['coef']) + fitted.get('output_bias',0.)
    if 'residual_mlp' in fitted:
        value=value.clamp(0.,1.)
        m=fitted['residual_mlp'];h=(x[:,m['columns']]-x.new_tensor(m['mean']))/x.new_tensor(m['scale'])
        for i,layer in enumerate(m['layers']):
            h=h@x.new_tensor(layer['weight']).T+x.new_tensor(layer['bias'])
            if i<len(m['layers'])-1:h=torch.tanh(h)
        value=value+h[:,0]
    return value.clamp(0., 1.)


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
