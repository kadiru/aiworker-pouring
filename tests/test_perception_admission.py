import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from perception_fit_data import admitted_scenes, admitted_tasks

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
    def test_failed_holdout_reuse_requires_explicit_history(self):
        plan=self.plan();old=dict(name='old_holdout',study_split='perception_test',calibration_only=True)
        plan['consumed_perception_training']=[old]
        with self.assertRaises(AssertionError):admitted_scenes(plan)
        plan['perception_history']=[dict(passed=False,consumed_training_names=['old_holdout'])]
        entries=admitted_scenes(plan)
        self.assertEqual(next(split for s,split,_ in entries if s['name']=='old_holdout'),'train')
        self.assertEqual(old['study_split'],'perception_test')
        plan['perception_holdout']=[old]
        with self.assertRaises(AssertionError):admitted_scenes(plan)

    def task_plan(self):
        plan=self.plan()
        plan['consumed_task_development']=[dict(name='dev0',study_split='development',policy_evaluation=True),
            dict(name='dev1',study_split='development',policy_evaluation=True)]
        plan['task_repair_history']=[dict(fit_splits={'dev0':'train','dev1':'validation'})]
        return plan

    def test_task_reuse_retains_recorded_identity_and_episode_split(self):
        plan=self.task_plan();original=copy.deepcopy(plan)
        entries=admitted_tasks(plan)
        self.assertEqual(plan,original)
        self.assertEqual([split for _,split,_ in entries],['train','validation'])
        self.assertTrue(all(s['study_split']=='development' for s,_,_ in entries))

    def test_task_reuse_rejects_reserved_or_still_active_ids(self):
        for part in ('development','test','perception_holdout','archived_perception_holdout'):
            plan=self.task_plan();plan[part]=[dict(name='dev0')]
            with self.assertRaises(AssertionError):admitted_tasks(plan)

    def test_task_reuse_requires_explicit_complete_admission(self):
        plan=self.task_plan();plan['task_repair_history']=[]
        with self.assertRaises(KeyError):admitted_tasks(plan)
        plan=self.task_plan();plan['consumed_task_development'].pop()
        with self.assertRaises(AssertionError):admitted_tasks(plan)
        plan=self.task_plan();plan['consumed_task_development'][0]['study_split']='test'
        with self.assertRaises(AssertionError):admitted_tasks(plan)

if __name__=='__main__':unittest.main()
