"""Native admission regressions: failed or mis-split records must stop fitting."""
import copy
import unittest
from unittest.mock import patch
import perception_repair_data as loader

class Admission(unittest.TestCase):
    def altered(self, filename, mutate):
        original=loader.read
        def read(path):
            value=original(path)
            if path.name==filename:
                value=copy.deepcopy(value);mutate(value)
            return value
        return patch.object(loader,'read',side_effect=read)

    def test_all_reviewed_cases_retained(self):
        with patch.object(loader,'protected_unchanged'):
            rows=loader.entries()
        self.assertEqual(len(rows),243)
        self.assertEqual(sum(split=='train' for spec,record,split in rows),207)

    def test_mislabeled_split_rejected(self):
        with patch.object(loader,'protected_unchanged'),self.altered('train-r2-audit.json',lambda a:a['records'][0].update(split='validation')):
            with self.assertRaises(AssertionError):loader.entries()

    def test_rejected_label_cannot_be_dropped(self):
        with patch.object(loader,'protected_unchanged'),self.altered('train-r2-audit.json',lambda a:a['records'][0]['label'].update(accepted=False)):
            with self.assertRaises(AssertionError):loader.entries()

    def test_missing_failed_scene_cannot_be_dropped(self):
        with patch.object(loader,'protected_unchanged'),self.altered('train-r2-audit.json',lambda a:a['records'].pop()):
            with self.assertRaises(AssertionError):loader.entries()

    def test_reserved_spec_change_rejected(self):
        with patch.object(loader,'protected_unchanged'),self.altered('protocol.json',lambda a:a['holdout'][0].update(name='changed-reserved')):
            with self.assertRaises(AssertionError):loader.entries()

    def test_loosened_accuracy_gate_rejected(self):
        with patch.object(loader,'protected_unchanged'),self.altered('protocol.json',lambda a:a['gates'].update(localization_max_error_mm=5)):
            with self.assertRaises(AssertionError):loader.entries()

if __name__=='__main__':unittest.main()
