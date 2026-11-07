import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
import pour_finish_study as study


class FinishResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(study,'ROOT',self.root);self.patch.start()
        self.addCleanup(self.patch.stop);self.addCleanup(self.tmp.cleanup)
        self.scenes=[dict(name=f'FN30_A_dev_{i:02}',cup_world=[.5,i*.01,1.04],initial_ml=30+i) for i in range(8)]
        (self.root/'frozen-plan.json').write_text(json.dumps(dict(development=self.scenes)))
        (self.root/'best.pt').write_bytes(b'unchanged-model')
        self.request=dict(kind='recording_infrastructure_resume',round=0,
            checkpoint_sha256=study.sha(self.root/'best.pt'),scene_plan_sha256=study.sha(self.root/'frozen-plan.json'),
            cases=[dict(planned_name='FN30_A_dev_05',name='FN30_A_dev_05_retry1_r0',retry_of='FN30_A_dev_05_r0'),
                   dict(planned_name='FN30_A_dev_06',name='FN30_A_dev_06_r0'),
                   dict(planned_name='FN30_A_dev_07',name='FN30_A_dev_07_r0')])
        (self.root/'infrastructure-aborts.json').write_text(json.dumps(dict(attempts={'FN30_A_dev_05_r0':{}})))
        self.save()

    def save(self):
        (self.root/'development-resume-plan.json').write_text(json.dumps(self.request))

    def test_only_interrupted_and_unstarted_cases_resume(self):
        with patch.object(study,'seen',return_value=[dict(name=f'FN30_A_dev_{i:02}_r0') for i in range(6)]):
            cases=study.development()
        self.assertEqual([s['name'] for s in cases],[c['name'] for c in self.request['cases']])
        for i,s in enumerate(cases,5):
            self.assertEqual(s['cup_world'],self.scenes[i]['cup_world'])
            self.assertEqual(s['initial_ml'],self.scenes[i]['initial_ml'])

    def test_started_retry_is_not_repeated(self):
        with patch.object(study,'seen',return_value=[dict(name=c['name']) for c in self.request['cases']]):
            self.assertEqual(study.development(),[])

    def test_retry_requires_catalogued_abort(self):
        (self.root/'infrastructure-aborts.json').write_text(json.dumps(dict(attempts={})))
        with patch.object(study,'seen',return_value=[]),self.assertRaises(AssertionError):study.development()

    def test_test_names_and_duplicate_cases_are_rejected(self):
        self.request['cases'][0]['name']='FN30_A_test_00';self.save()
        with patch.object(study,'seen',return_value=[]),self.assertRaises(AssertionError):study.development()
        self.request['cases']=[self.request['cases'][1]]*2;self.save()
        with patch.object(study,'seen',return_value=[]),self.assertRaises(AssertionError):study.development()

    def test_frozen_or_changed_controller_cannot_resume(self):
        (self.root/'best.pt').write_bytes(b'changed')
        with self.assertRaises(AssertionError):study.development()
        (self.root/'final-selection.json').write_text('{}')
        with self.assertRaises(AssertionError):study.development()


if __name__=='__main__':unittest.main()
