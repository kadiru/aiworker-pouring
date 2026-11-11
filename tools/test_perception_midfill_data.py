"""Reject split leakage, missing review, changed data and failed native records."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import perception_midfill_data as data


class MidfillAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.base = [(dict(name=f'old{i}'), dict(sha256=f'old-sha{i}'), 'train' if i < 207 else 'validation') for i in range(243)]
        self.specs = [dict(name=f'new{i}', study_split='train') for i in range(36)]
        self.records = [dict(name=s['name'], split='train', path='/native/'+s['name'], sha256=s['name'],
                             recording_validation=dict(valid=True), label=dict(accepted=True), containment=dict(final=0, ever_table=0)) for s in self.specs]
        self.files = {'protocol.json': dict(train=self.specs, midfill_train_names=[s['name'] for s in self.specs], holdout=[dict(name='reserved')]),
                      'plan.json': dict(holdout=[dict(name='old_reserved')]),
                      'train-r3-audit.json': dict(complete=True, records=self.records),
                      'train-r3-manual-review.json': dict(accepted=True, audit_sha256='train-r3-audit.json')}
        self.patches = [patch.object(data, 'protected_unchanged'),
                        patch.object(data, 'previous_entries', side_effect=lambda: copy.deepcopy(self.base)),
                        patch.object(data, 'read', side_effect=lambda p: self.files[Path(p).name]),
                        patch.object(data, 'sha', side_effect=lambda p: Path(p).name)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)

    def test_all_required_cases_kept(self):
        rows = data.entries()
        self.assertEqual(len(rows), 279)
        self.assertEqual(sum(split == 'validation' for s, r, split in rows), 36)

    def test_reserved_overlap_rejected(self):
        self.files['protocol.json']['holdout'].append(dict(name='new0'))
        with self.assertRaises(AssertionError): data.entries()

    def test_validation_relabel_rejected(self):
        self.specs[0]['study_split'] = 'validation'
        with self.assertRaises(AssertionError): data.entries()

    def test_duplicate_episode_rejected(self):
        self.base[0][0]['name'] = 'new0'
        with self.assertRaises(AssertionError): data.entries()

    def test_missing_review_rejected(self):
        self.files['train-r3-manual-review.json']['accepted'] = False
        with self.assertRaises(AssertionError): data.entries()

    def test_changed_source_rejected(self):
        self.records[0]['sha256'] = 'changed'
        with self.assertRaises(AssertionError): data.entries()

    def test_loss_or_bad_label_rejected(self):
        for field in ('loss', 'label', 'recording'):
            original = copy.deepcopy(self.records[0])
            if field == 'loss': self.records[0]['containment']['final'] = 1
            elif field == 'label': self.records[0]['label']['accepted'] = False
            else: self.records[0]['recording_validation']['valid'] = False
            with self.assertRaises(AssertionError): data.entries()
            self.records[0] = original


if __name__ == '__main__':
    unittest.main()
