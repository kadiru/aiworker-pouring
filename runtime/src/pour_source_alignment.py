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
