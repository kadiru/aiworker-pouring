import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
from pour_camera_clock import camera_time_is_current,recording_tolerance


class CameraClockTests(unittest.TestCase):
    def test_observed_roundoff_is_accepted(self):
        self.assertTrue(camera_time_is_current(44.754998999647796,44.755))

    def test_old_physics_and_control_frames_are_rejected(self):
        for lag in (.001,.0025,.05,.1):
            self.assertFalse(camera_time_is_current(44.755-lag,44.755))

    def test_future_frames_and_nonfinite_times_are_rejected(self):
        self.assertFalse(camera_time_is_current(44.7575,44.755))
        for value in (math.nan,math.inf,-math.inf):
            self.assertFalse(camera_time_is_current(value,44.755))
            self.assertFalse(camera_time_is_current(44.755,value))

    def test_legacy_precision_is_preserved(self):
        self.assertEqual(recording_tolerance({}),1e-6)
        self.assertFalse(camera_time_is_current(44.754998999647796,44.755,recording_tolerance({})))

    def test_recording_cannot_widen_the_bound(self):
        self.assertEqual(recording_tolerance({'camera_timestamp_tolerance_s':1e-4}),1e-4)
        for value in (0,-1,.001,math.inf,math.nan):
            with self.assertRaises(AssertionError):recording_tolerance({'camera_timestamp_tolerance_s':value})


if __name__=='__main__':unittest.main()
