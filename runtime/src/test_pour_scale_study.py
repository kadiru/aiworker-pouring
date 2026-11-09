import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pour_scale_study as study


class ScaleAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(study, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.root/'best.pt').write_bytes(b'test checkpoint, not learned weights')
        self.scene = dict(name='SC_A_TEST_ONLY', calibration_only=True, policy_evaluation=False)
        self.plan = dict(pilot=[self.scene], collect=[self.scene], holdout=[self.scene])
        self.save()

    def save(self):
        (self.root/'plan.json').write_text(json.dumps(self.plan))
        self.manifest = dict(checkpoint_sha256=study.sha(self.root/'best.pt'),
                             plan_sha256=study.sha(self.root/'plan.json'),
                             runtime_sources={}, protected_parent_files={}, seed_and_camera_accepted=False)
        self.write_manifest()

    def write_manifest(self):
        (self.root/'candidate.json').write_text(json.dumps(self.manifest))

    def test_autonomous_spec_cannot_enter_calibration_queue(self):
        self.scene['policy_evaluation'] = True
        self.save()
        with self.assertRaises(AssertionError): study.pending('pilot')

    def test_collection_requires_reviewed_seed_and_camera(self):
        with self.assertRaises(AssertionError): study.pending('collect')

    def test_changed_parent_blocks_pilot(self):
        parent=self.root/'parent.json';parent.write_text('original')
        self.manifest['protected_parent_files']={str(parent):study.sha(parent)}
        self.write_manifest();parent.write_text('changed')
        with self.assertRaises(AssertionError): study.pending('pilot')

    def test_holdout_requires_identical_selection(self):
        selection=dict(self.manifest,all_size_component_gates_passed=True,plan_sha256='different')
        (self.root/'component-selection.json').write_text(json.dumps(selection))
        with self.assertRaises(AssertionError): study.pending('holdout')


if __name__ == '__main__': unittest.main()
