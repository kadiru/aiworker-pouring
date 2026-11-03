"""CPU-only checks of the actual recovered retargeter, without loading Kit."""
import ast
from dataclasses import dataclass
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

@dataclass
class RetargeterCfg:
    sim_device: str = "cpu"
class RetargeterBase:
    def __init__(self, cfg): self._sim_device = cfg.sim_device

source = Path('/workspace/robotis_lab/source/robotis_lab/robotis_lab/devices/quest3_controller.py')
tree = ast.parse(source.read_text())
names = {'Quest3ControllerRelRetargeterCfg', 'Quest3ControllerRelRetargeter'}
scope = dict(np=np, torch=torch, Rotation=Rotation, dataclass=dataclass,
             RetargeterCfg=RetargeterCfg, RetargeterBase=RetargeterBase,
             CONTROLLER_RIGHT='controller_right', POSE_ROW=0, INPUT_ROW=1, TRIGGER=2, SQUEEZE=3)
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names], type_ignores=[]), str(source), 'exec'), scope)

class Tests(unittest.TestCase):
    def setUp(self):
        self.r = scope['Quest3ControllerRelRetargeter'](scope['Quest3ControllerRelRetargeterCfg']())
    def sample(self, x=0.0, trigger=1, squeeze=0):
        return np.array([[x,0,1,1,0,0,0],[0,0,trigger,squeeze,0,0,0]], dtype=np.float32)
    def apply(self, sample): return self.r.retarget({'controller_right':sample}).numpy()
    def test_deadman_and_reacquire(self):
        self.assertTrue(np.all(self.apply(self.sample())[:6] == 0))
        self.assertGreater(self.apply(self.sample(.01))[0],0)
        self.assertTrue(np.all(self.apply(self.sample(.5,trigger=0))[:6] == 0))
        self.assertTrue(np.all(self.apply(self.sample(.6))[:6] == 0))
    def test_tracking_loss_resets_reference(self):
        self.apply(self.sample())
        self.assertTrue(np.all(self.r.retarget({}).numpy()[:6] == 0))
        self.assertTrue(np.all(self.apply(self.sample(.7))[:6] == 0))
    def test_invalid_packets_hold(self):
        for sample in (np.full((2,7),np.nan),np.zeros((2,7)),np.zeros((3,7))):
            self.apply(self.sample())
            self.assertTrue(np.all(self.apply(sample)[:6] == 0))
            self.assertTrue(np.all(self.apply(self.sample(.8))[:6] == 0))
    def test_clamp_and_gripper(self):
        self.apply(self.sample())
        a=self.apply(self.sample(10,squeeze=1))
        self.assertLessEqual(abs(float(a[0])),.080001)
        self.assertEqual(a[6],-1)
        self.assertEqual(self.apply(self.sample(10,trigger=0,squeeze=0))[6],1)
    def test_quaternion_double_cover(self):
        self.apply(self.sample())
        sample=self.sample();sample[0,3]=-1
        self.assertTrue(np.allclose(self.apply(sample)[3:6],0))

if __name__ == '__main__': unittest.main(verbosity=2)
