"""Use retained native failures to check explicit benchmark selection limits."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from wide_release_gate import validate_development_outcome


class ReleaseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r2=json.loads((ROOT/'evidence/wide-A/development-r2-progress.json').read_text())['rows']
        cls.r1=json.loads((ROOT/'evidence/wide-A/development-r1-progress.json').read_text())['rows']

    def check(self,row,reason=None):
        validate_development_outcome(row,row['strict_pass'],row['failure_reasons'],reason)

    def test_clean_completed_case_needs_no_exception(self):
        self.check(next(r for r in self.r2 if r['strict_pass']))

    def test_retained_underfill_requires_explicit_rationale(self):
        row=next(r for r in self.r2 if 'underfilled' in r['failure_reasons'])
        with self.assertRaises(AssertionError):self.check(row)
        self.check(row,'Evaluate the known conservative full-stop limit with unchanged scores.')
        self.assertFalse(row['strict_pass'])

    def test_unconfirmed_full_stays_a_failure(self):
        row=next(r for r in self.r2 if r['stop_reason']=='fill_unconfirmed')
        self.check(row,'Evaluate the known full-stop confirmation limit on unused scenes.')
        self.assertFalse(row['strict_pass'])

    def test_recorded_spill_cannot_be_excused(self):
        row=next(r for r in self.r1 if r['final_outside_ml']>0)
        with self.assertRaises(AssertionError):self.check(row,'Evaluate known conservative full-stop limits only.')

    def test_physical_failure_cannot_hide_under_an_incomplete_reason_list(self):
        source=next(r for r in self.r2 if 'underfilled' in r['failure_reasons'])
        for key,value in [('grasp_verified',False),('final_outside_ml',.005832),
                          ('completed_return_release_sequence',False),('teacher_fallback',True)]:
            row=copy.deepcopy(source);row[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(row,'Evaluate known conservative full-stop limits only.')


if __name__=='__main__':unittest.main()
