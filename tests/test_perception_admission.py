import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from perception_fit_data import admitted_scenes

class Admission(unittest.TestCase):
    def plan(self):
        return dict(collect=[dict(name='train',study_split='train'),dict(name='val',study_split='validation')],
            level_supplement=[dict(name='extra',study_split='train')],camera_pilot=[dict(name='pilot',study_split='camera_pilot',calibration_only=True)],
            perception_training_pilots=['pilot'],test=[dict(name='reserved')])
    def test_explicit_reuse_keeps_original_recorded_split(self):
        plan=self.plan();original=copy.deepcopy(plan);entries=admitted_scenes(plan)
        self.assertEqual(plan,original)
        self.assertEqual(entries[-1][0]['study_split'],'camera_pilot')
        self.assertEqual(entries[-1][1],'train')
        self.assertEqual([s['name'] for s,split,_ in entries if split=='validation'],['val'])
    def test_rejects_reserved_admission(self):
        plan=self.plan();plan['level_supplement'][0]['name']='reserved'
        with self.assertRaises(AssertionError):admitted_scenes(plan)
    def test_rejects_validation_record_disguised_as_pilot(self):
        plan=self.plan();plan['camera_pilot'][0]['study_split']='validation'
        with self.assertRaises(AssertionError):admitted_scenes(plan)

if __name__=='__main__':unittest.main()
