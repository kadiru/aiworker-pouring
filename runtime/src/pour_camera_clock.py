"""Bound raw camera-clock roundoff without admitting an old physics frame."""
import math

LEGACY_TOLERANCE_S = 1e-6
CAMERA_TOLERANCE_S = 1e-4


def camera_time_is_current(camera_time, physics_time, tolerance=CAMERA_TOLERANCE_S):
    return (math.isfinite(camera_time) and math.isfinite(physics_time)
            and 0 < tolerance <= CAMERA_TOLERANCE_S
            and abs(camera_time-physics_time) < tolerance)


def recording_tolerance(attributes):
    value = float(attributes.get('camera_timestamp_tolerance_s', LEGACY_TOLERANCE_S))
    assert math.isfinite(value) and 0 < value <= CAMERA_TOLERANCE_S
    return value
