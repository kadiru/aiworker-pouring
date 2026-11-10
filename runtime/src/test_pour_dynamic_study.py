"""Synthetic authorization-gate checks; these are not simulation outcomes."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pour_dynamic_study as study

class StudyGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.patch=patch.object(study,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.m=dict(runtime_sources={},protected_parent_files={},high_seeds_accepted=True,autonomy_accepted=False)
        self.cal=dict(name='SD_A_synthetic_calibration',calibration_only=True,policy_evaluation=False)
        self.auto=dict(name='SD_A_synthetic_autonomy',policy_evaluation=True)
        self.p=dict(holdout=[self.cal],autonomy=[self.auto]);self.write('candidate.json',self.m);self.write('protocol.json',self.p)
    def write(self,n,v):(self.root/n).write_text(json.dumps(v))
    def selection(self,**kwargs):return dict(protocol_sha256=study.sha(self.root/'protocol.json'),frozen_files={},**kwargs)
    def geometry(self,part,s):self.write(part+'-geometry.json',dict(passed=True,complete=True,protocol_sha256=study.sha(self.root/'protocol.json'),cases=[dict(name=s['name'])]))
    def test_unaccepted_color_keeps_reserved_cases_unused(self):
        self.write('component-selection.json',self.selection(all_required_development_gates_passed=False))
        with self.assertRaises(AssertionError):study.pending('holdout')
    def test_component_success_does_not_enable_autonomy(self):
        self.write('autonomy-selection.json',self.selection(all_gates_passed=True))
        with self.assertRaises(AssertionError):study.pending('autonomy')
    def test_model_change_invalidates_freeze(self):
        (self.root/'synthetic.pt').write_bytes(b'before');s=self.selection(all_required_development_gates_passed=True)
        s['frozen_files']={'synthetic.pt':study.sha(self.root/'synthetic.pt')};self.write('component-selection.json',s)
        (self.root/'synthetic.pt').write_bytes(b'after')
        with self.assertRaises(AssertionError):study.pending('holdout')
    def test_calibration_cannot_masquerade_as_autonomy(self):
        self.m['autonomy_accepted']=True;self.write('candidate.json',self.m)
        self.p['autonomy']=[self.cal];self.write('protocol.json',self.p)
        self.write('autonomy-selection.json',self.selection(all_gates_passed=True))
        with self.assertRaises(AssertionError):study.pending('autonomy')
    def test_frozen_synthetic_calibration_is_admitted(self):
        self.write('component-selection.json',self.selection(all_required_development_gates_passed=True));self.geometry('holdout',self.cal)
        self.assertEqual(study.pending('holdout'),[self.cal])
if __name__=='__main__':unittest.main()
