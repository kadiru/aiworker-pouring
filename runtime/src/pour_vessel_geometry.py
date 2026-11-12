"""Declared cup dimensions for the first size curriculum (metres).

This is fixture/configuration information, not a visual size estimator.
"""
import math
import numpy as np


def cup_profile(spec):
    from bottle_pour import CUP
    h = float(spec.get('cup_height_scale', 1.))
    d = float(spec.get('cup_diameter_scale', 1.))
    if not (math.isfinite(h) and math.isfinite(d) and .9 <= h <= 1.1 and .9 <= d <= 1.1):
        raise ValueError('Cup curriculum permits independent height/diameter scales in [0.9, 1.1]')
    from pour_bottle_geometry import profile_for
    profile_for(spec)
    # Keep the physical base thickness constant; vary usable internal height.
    bottom = CUP[0][0]
    return tuple((z if h == 1. else bottom + (z-bottom)*h, r*d) for z, r in CUP)


def capacity_ml(profile):
    from bottle_pour import SIDES, CUP, CAPACITY_ML
    if tuple(profile) == CUP:
        return CAPACITY_ML
    factor = SIDES*math.sin(2*math.pi/SIDES)/6*1e6
    return factor*sum((b[0]-a[0])*(a[1]**2+a[1]*b[1]+b[1]**2) for a,b in zip(profile, profile[1:]))


def cup_geometry(spec):
    from bottle_pour import CUP, BRIM_NOMINAL_ML
    p = cup_profile(spec)
    return dict(bottom_z_m=p[0][0], rim_z_m=p[-1][0], rim_radius_m=p[-1][1],
        profile=[list(v) for v in p], capacity_geometric_ml=capacity_ml(p),
        # Legacy inventory reference only: this ratio is NOT a fluid calibration.
        legacy_brim_nominal_ml=BRIM_NOMINAL_ML*(capacity_ml(p)/capacity_ml(CUP)),
        source='declared_fixture_geometry_not_visual_estimation')


def initial_cup_points(spec):
    """Sparse, bottom-up particle grid; reject overfill before creating the scene."""
    from bottle_pour import SIDES
    p = cup_profile(spec); spacing = float(spec['spacing'])
    count = round(float(spec.get('cup_initial_ml', 0.))/(spacing**3*1e6))
    if count < 0 or spacing <= 0:
        raise ValueError('Invalid initial inventory or spacing')
    radius = p[0][1]*math.cos(math.pi/SIDES)-2*spacing
    n = int(radius/spacing)
    cross = [(i*spacing,j*spacing) for i in range(-n,n+1) for j in range(-n,n+1)
             if (i*i+j*j)*spacing**2 < radius**2]
    if not cross:
        raise ValueError('Cup is too narrow at this particle spacing')
    points = np.array([[*cross[i%len(cross)],p[0][0]+2*spacing+(i//len(cross))*spacing]
                       for i in range(count)], dtype=np.float32).reshape(-1,3)
    if len(points) and points[:,2].max() >= p[-1][0]-spacing:
        raise ValueError('Initial grid exceeds cup rim; reduce calibration inventory')
    return points


def settled_cup_points(spec, seed):
    """Select a low settled layer without stretching particle separations."""
    from bottle_pour import SIDES
    p=cup_profile(spec);spacing=float(spec['spacing'])
    seed=np.asarray(seed,dtype=np.float32)
    if seed.ndim!=2 or seed.shape[1]!=3 or not np.isfinite(seed).all():
        raise ValueError('Malformed settled particle seed')
    z=seed[:,2];angle=(np.arctan2(seed[:,1],seed[:,0])%(2*math.pi/SIDES))-math.pi/SIDES
    radius=np.interp(z,[v[0] for v in p],[v[1] for v in p])*math.cos(math.pi/SIDES)/np.cos(angle)
    keep=(z>=p[0][0])&(z<p[-1][0]-2*spacing)&(np.linalg.norm(seed[:,:2],axis=1)<radius-.5*spacing)
    candidates=seed[keep]
    count=round(spec['cup_initial_ml']/(spacing**3*1e6))
    if count<1 or len(candidates)<count:
        raise ValueError('Settled seed does not cover requested cup inventory')
    order=np.argsort(candidates[:,2],kind='stable')
    return candidates[order[:count]].copy()
