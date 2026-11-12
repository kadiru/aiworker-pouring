"""Declared geometry and authored shell mass properties for the bottle curriculum."""
import math

import numpy as np


BOTTLE = ((.004, .025), (.082, .025), (.107, .009), (.135, .009))
WALL = .0024
GRASP_Z = .048
FIXED_BAND_TOP = .065
SIDES = 20
BASE_MASS = .18


def resized_profile(height_scale, diameter_scale):
    h, d = float(height_scale), float(diameter_scale)
    if not (math.isfinite(h) and math.isfinite(d) and .85 <= h <= 1.15 and .9 <= d <= 1.):
        raise ValueError("Bottle height must be in [0.85, 1.15] and diameter in [0.90, 1.00]")
    lip = BOTTLE[-1][0]
    ratio = (lip*h-FIXED_BAND_TOP)/(lip-FIXED_BAND_TOP)
    return tuple((z if z <= FIXED_BAND_TOP else FIXED_BAND_TOP+(z-FIXED_BAND_TOP)*ratio,
                  r*d if r == BOTTLE[0][1] else r) for z, r in BOTTLE)


def profile_for(spec):
    height = spec.get("bottle_height_scale", 1.)
    diameter = spec.get("bottle_diameter_scale", 1.)
    if spec.get("bottle_neck_scale", 1.) != 1.:
        raise ValueError("Bottle mouth diameter must remain at its calibrated size")
    if spec.get("bottle_size_autonomy"):
        required = ("bottle", "robot", "policy_evaluation", "size_autonomy_prefill")
        if not all(spec.get(k) for k in required):
            raise ValueError("Resized bottles require a size-aware autonomous robot fixture")
        return resized_profile(height, diameter)
    if height != 1. or diameter != 1.:
        raise ValueError("Bottle scaling requires the size-aware autonomy fixture")
    return BOTTLE


def profile_volume(profile):
    factor = SIDES*math.sin(2*math.pi/SIDES)/6
    return factor*sum((b[0]-a[0])*(a[1]**2+a[1]*b[1]+b[1]**2)
                      for a,b in zip(profile, profile[1:]))


def shell_volume(profile):
    outer = tuple((z,r+WALL) for z,r in profile)
    return profile_volume(outer)-profile_volume(profile)+math.pi*outer[0][1]**2*outer[0][0]


def moments(profile, thickness=WALL, sides=SIDES, density=1.):
    """Integrate the circular base and tapered regular-polygon shell."""
    p = np.asarray(profile, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("Invalid profile")
    if not (p>0).all() or not (np.diff(p[:,0])>0).all() or thickness<=0 or density<=0 or sides<3:
        raise ValueError("Invalid shell dimensions or density")
    z0, r0 = p[0]
    radius = r0 + thickness
    volume = math.pi * radius**2 * z0
    first = volume*z0/2
    second = volume*z0**2/3
    polar = volume*radius**2/2
    area_factor = sides*math.sin(2*math.pi/sides)/2
    polar_factor = sides*math.sin(2*math.pi/sides)*(2+math.cos(2*math.pi/sides))/12
    # Three Gauss nodes integrate the volume and degree-four moment terms.
    nodes, weights = np.polynomial.legendre.leggauss(3)
    for (a,ra), (b,rb) in zip(p,p[1:]):
        z = (a+b)/2 + nodes*(b-a)/2
        w = weights*(b-a)/2
        r = ra+(rb-ra)*(z-a)/(b-a)
        area = area_factor*((r+thickness)**2-r**2)
        q = polar_factor*((r+thickness)**4-r**4)
        volume += float(w@area)
        first += float(w@(area*z))
        second += float(w@(area*z*z))
        polar += float(w@q)
    center = first/volume
    transverse = density*(polar/2+second-volume*center*center)
    inertia = [transverse, transverse, density*polar]
    assert all(v>0 for v in inertia) and 0<center<p[-1,0]
    return dict(volume_m3=volume,mass_kg=volume*density,
                center_of_mass_m=[0.,0.,center],diagonal_inertia_kg_m2=inertia)


def declared_geometry(spec):
    profile = profile_for(spec)
    return dict(profile=[list(ring) for ring in profile], lip_height_m=profile[-1][0],
                outer_body_diameter_m=2*(profile[0][1]+WALL),
                grasp_height_m=GRASP_Z, capacity_geometric_ml=profile_volume(profile)*1e6,
                **moments(profile, density=BASE_MASS/shell_volume(BOTTLE)))
