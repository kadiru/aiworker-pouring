"""Undeployed bottle curriculum geometry; never changes runtime globals.

Height variation starts above the fixed grasp band. Width only decreases;
mouth, wall thickness, bottom and grasp height keep their baseline values.
These calculations are not native collision/grasp/inertia acceptance.
"""
import math
BASE = ((.004, .025), (.082, .025), (.107, .009), (.135, .009))
WALL = .0024
GRASP_Z = .048
FIXED_BAND_TOP = .065
SIDES = 20


def profile(height_scale=1., body_scale=1.):
    h, w = float(height_scale), float(body_scale)
    if not (math.isfinite(h) and math.isfinite(w) and .85 <= h <= 1.15 and .9 <= w <= 1.):
        raise ValueError('Only height [.85,1.15] and body radius [.9,1] are proposed')
    ratio = (.135*h-FIXED_BAND_TOP)/(.135-FIXED_BAND_TOP)
    source = (BASE[0], (FIXED_BAND_TOP, .025), *BASE[1:])
    return tuple((z if z <= FIXED_BAND_TOP else FIXED_BAND_TOP+(z-FIXED_BAND_TOP)*ratio,
                  r*w if r == .025 else r) for z, r in source)


def volume(p):
    factor = SIDES*math.sin(2*math.pi/SIDES)/6
    return factor*sum((b[0]-a[0])*(a[1]**2+a[1]*b[1]+b[1]**2) for a,b in zip(p,p[1:]))


def shell_volume(p):
    # Wall sectors use regular polygons; the native base is a circular cylinder.
    return volume(tuple((z,r+WALL) for z,r in p))-volume(p)+math.pi*(p[0][1]+WALL)**2*p[0][0]


def geometry(h, w):
    p=profile(h,w)
    return dict(height_scale=h,body_scale=w,profile=[list(v) for v in p],
        lip_height_m=p[-1][0],outer_body_diameter_m=2*(p[0][1]+WALL),
        inner_mouth_diameter_m=2*p[-1][1],grasp_height_m=GRASP_Z,
        grasp_band_m=[.031,.065],wall_m=WALL,capacity_geometric_ml=volume(p)*1e6,
        proposed_empty_mass_kg=.18*shell_volume(p)/shell_volume(BASE),
        mass_policy='Constant authored shell density relative to baseline 0.18 kg; not physical calibration; native inertia still required',
        native_grasp_accepted=False,native_inertia_accepted=False,autonomy_accepted=False)
