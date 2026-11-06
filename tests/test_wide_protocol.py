"""Protect fresh-test identities and immutable selection from historical data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'runtime/src/pour_wide_study.py'
spec=importlib.util.spec_from_file_location('wide_protocol',SOURCE)
wide=importlib.util.module_from_spec(spec);spec.loader.exec_module(wide)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'outputs';self.src=self.root/'src';self.src.mkdir()
        for name,value in [('ROOT',self.root),('OUTPUTS',self.output),('SOURCES',self.src)]:
            p=patch.object(wide,name,value);p.start();self.addCleanup(p.stop)

    def record(self,name,protocol=17):
        p=self.output/'run'/name/'parameters.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(name=name,protocol=protocol)))

    def test_historical_t30_tests_and_old_protocol_are_excluded(self):
        self.record('T30_A_test_00');self.record('T30_A_demo_train_00',16)
        self.record('T30_A_demo_train_01');self.record('XW30_A_test_02')
        self.assertEqual({s['name'] for s in wide.seen()},{'T30_A_demo_train_01','XW30_A_test_02'})

    def freeze(self):
        p=self.root/'frozen-plan.json';p.write_text(json.dumps(dict(test=[dict(name=f'XW30_A_test_{i:02}') for i in range(3)])))
        model=self.root/'best.pt';model.write_bytes(b'test checkpoint, not a model')
        runtime=self.src/'controller.py';runtime.write_text('x=1\n')
        sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
        (self.root/'final-selection.json').write_text(json.dumps(dict(selected_checkpoint_sha256=sha(model),
            scene_plan_sha256=sha(p),runtime_sources={'controller.py':sha(runtime)})))

    def test_partial_attempt_is_not_retried(self):
        self.freeze();self.record('XW30_A_test_01');self.record('T30_A_test_00')
        self.assertEqual([s['name'] for s in wide.tests()],['XW30_A_test_00','XW30_A_test_02'])

    def test_frozen_source_model_or_plan_change_blocks_tests(self):
        for name in ['src/controller.py','best.pt','frozen-plan.json']:
            with self.subTest(name=name):
                self.freeze();p=self.root/name;p.write_bytes(p.read_bytes()+b'\n')
                with self.assertRaises(AssertionError):wide.tests()

    def test_no_more_references_or_development_after_freeze(self):
        self.freeze()
        for fn in (wide.references,wide.repairs,wide.development):
            with self.assertRaises(AssertionError):fn()

    def test_candidates_are_exact_swap_pairs_with_balanced_stop_targets(self):
        pair=next(wide.candidates({},'test',202611054))
        def positions(s):
            x,y,z=s['cup_world'];a,b=s['source_xy']
            return (x,y),(x-b,y+a)
        a,b=map(positions,pair)
        for x,y in zip(a[0]+a[1],b[1]+b[0]):self.assertAlmostEqual(x,y)
        self.assertEqual({s['expected_stop'] for s in pair},{'cup_near_full','bottle_empty'})
        self.assertTrue(all(s['policy_evaluation'] and not s['expert_recovery'] for s in pair))


if __name__=='__main__':unittest.main()
