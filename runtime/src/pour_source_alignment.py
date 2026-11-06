"""Normalize the fixed source-camera crop using the RGB-localized cup.

The pouring mouth follows the cup. Returning its image to the original training
frame removes cup-position translation; wrist RGB stays in its attached frame.
Only stored calibration and visual cup XY are used, never live scene poses.
"""
import numpy as np
from math import isfinite
from PIL import Image
from pour_guard_vision import project
from pour_stop_data import crops


def aligned_crops(task,wrist,cup_xy,cal):
    z=cal['table_z']+.038
    current=project([*cup_xy,z],cal)
    reference=cal.get('source_reference_calibration')
    if reference is not None:
        nominal=project([.49,-.15,z],reference)
        right=np.asarray(cal['T'])[:3,0]
        ref_right=np.asarray(reference['T'])[:3,0]
        radius=np.linalg.norm(project(np.r_[cup_xy,z]+right*.01,cal)-current)
        ref_radius=np.linalg.norm(project(np.array([.49,-.15,z])+ref_right*.01,reference)-nominal)
        scale=float(radius/ref_radius)
        image=Image.fromarray(task).transform((384,384),Image.Transform.AFFINE,
            (scale,0.,float(current[0]-scale*nominal[0]),0.,scale,float(current[1]-scale*nominal[1])),
            Image.Resampling.BILINEAR,fillcolor=(180,180,180))
        return crops(np.asarray(image).copy(),wrist)
    nominal=project([.49,-.15,z],cal)
    du,dv=current-nominal
    image=Image.fromarray(task).transform((384,384),Image.Transform.AFFINE,
        (1.,0.,float(du),0.,1.,float(dv)),Image.Resampling.BILINEAR,fillcolor=(180,180,180))
    return crops(np.asarray(image).copy(),wrist)


def combine_evidence(raw_empty,raw_present,aligned_empty,aligned_present):
    """Require presence in the same view supplying empty evidence."""
    views=[(e,p) for e,p in ((raw_empty,raw_present),(aligned_empty,aligned_present))
        if isfinite(e) and isfinite(p) and 0<=e<=1 and 0<=p<=1]
    return (max((e for e,p in views if p>=.7),default=0.),
        max((p for _,p in views),default=0.))


def held_source_presence(raw_present,aligned_present,encoder_hold):
    """Either camera framing may see the bottle; fingers must still hold it."""
    if not encoder_hold:return 0.
    return max((p for p in (raw_present,aligned_present) if isfinite(p) and 0<=p<=1),default=0.)
