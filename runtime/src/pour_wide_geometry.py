"""Fixed vessel-clearance geometry for the wider-table controller.

Uses only known vessel geometry and an RGB-localized starting displacement.
This is an engineered pose constraint, not a learned component.
"""
from bottle_pour import pose, ease


def calibrated_pose(angle, lift, start_xy, mouth_z, centered=False):
    # Keep the bottle's bottom above the receiving rim while it crosses the
    # cup in swapped layouts. The old height boost disappeared too early.
    # Four extra millimetres at the lip cover wall/contact-offset clearance.
    start = [0., 0.] if centered else start_xy
    base, rotation = pose(angle, lift, start, mouth_z + .004)
    base[2] += .018 * lift * (1. - ease((angle - 45.) / 25.))
    return base, rotation
