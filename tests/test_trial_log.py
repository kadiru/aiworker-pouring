import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trial_log", ROOT / "tools/trial_log.py")
log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log)


class TrialLogTests(unittest.TestCase):
    def setUp(self):
        self.r = json.loads((ROOT / "examples/synthetic-wet.json").read_text())

    def test_correct_mass_accounting(self):
        result = log.evaluate(self.r)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["delivered_g"], 50)
        self.assertEqual(result["metrics"]["mass_balance_error_g"], 0)

    def test_acceptance_boundaries(self):
        for delivered, expected in ((44.9, False), (45, True), (55, True), (55.1, False)):
            r = copy.deepcopy(self.r)
            r["cup_after_g"] = r["cup_before_g"] + delivered
            r["bottle_after_g"] = r["bottle_before_g"] - delivered - 1
            self.assertEqual(log.evaluate(r)["passed"], expected)

    def test_spill_and_mass_balance_failures(self):
        self.r["spill_after_g"] = 103
        self.assertFalse(log.evaluate(self.r)["passed"])
        self.r["spill_after_g"] = 101
        self.r["bottle_after_g"] = 120
        self.assertIn("mass_accounting_invalid", log.evaluate(self.r)["reasons"])

    def test_unmeasured_aborted_attempt_is_retained_as_failure(self):
        self.r["cycle_complete"] = False
        self.r["cup_after_g"] = None
        self.assertFalse(log.evaluate(self.r)["passed"])
        self.assertEqual(log.summarize([self.r])["groups"][0]["attempts"], 1)

    def test_dry_has_no_mass_claim(self):
        self.r["phase"] = "dry"
        with self.assertRaises(ValueError):
            log.validate(self.r)
        for k in log.MASSES:
            self.r[k] = None
        self.assertTrue(log.evaluate(self.r)["passed"])
        self.assertEqual(log.evaluate(self.r)["metrics"], {})

    def test_invalid_numbers_and_flags(self):
        for value in (True, -1, float("nan"), float("inf"), "50"):
            r = copy.deepcopy(self.r)
            r["cup_after_g"] = value
            with self.assertRaises(ValueError):
                log.validate(r)
        self.r["collision"] = "false"
        with self.assertRaises(ValueError):
            log.validate(self.r)

    def test_timestamp_and_learned_checkpoint_required(self):
        self.r["started_at"] = "2026-10-28T10:00:00"
        with self.assertRaises(ValueError):
            log.validate(self.r)
        self.r["started_at"] += "-07:00"
        self.r["mode"] = "learned"
        with self.assertRaises(ValueError):
            log.validate(self.r)

    def test_json_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            log.decode('{"collision": true, "collision": false}')
        with self.assertRaises(ValueError):
            log.decode('{"duration_s": NaN}')

    def test_synthetic_cannot_meet_physical_gate(self):
        rows = []
        for i in range(5):
            r = copy.deepcopy(self.r)
            r["trial_id"] = str(i)
            rows.append(r)
        result = log.summarize(rows)
        self.assertEqual(result["physical_trials"], 0)
        self.assertIsNone(result["groups"][0]["meets_physical_five_trial_rule"])

    def test_incident_prevents_physical_gate_even_with_four_passes(self):
        # In-memory validation fixture only; never written as physical evidence.
        rows = []
        for i in range(5):
            r = copy.deepcopy(self.r)
            r.update(trial_id=str(i), evidence_kind="physical")
            rows.append(r)
        rows[-1]["cup_after_g"] = 120
        self.assertTrue(log.summarize(rows)["groups"][0]["meets_physical_five_trial_rule"])
        rows[-1]["collision"] = True
        self.assertFalse(log.summarize(rows)["groups"][0]["meets_physical_five_trial_rule"])

    def test_conditions_are_separate(self):
        r2 = copy.deepcopy(self.r)
        r2.update(trial_id="2", evidence_kind="simulation", series_id="SIM")
        self.assertEqual(len(log.summarize([self.r, r2])["groups"]), 2)

    def test_duplicate_and_changed_series_rejected_without_mutating_log(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evidence") as d:
            p = Path(d) / "trials.jsonl"
            log.record(p, self.r)
            before = p.read_bytes()
            with self.assertRaises(ValueError):
                log.record(p, self.r)
            self.r.update(trial_id="2", mode="teleoperated")
            with self.assertRaises(ValueError):
                log.record(p, self.r)
            self.assertEqual(p.read_bytes(), before)

    def test_corrupt_log_is_not_appended(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evidence") as d:
            p = Path(d) / "trials.jsonl"
            p.write_text('{"truncated":')
            with self.assertRaises(ValueError):
                log.record(p, self.r)
            self.assertEqual(p.read_text(), '{"truncated":')

    def test_cli_end_to_end_and_duplicate_exit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evidence") as d:
            p = Path(d) / "trials.jsonl"
            cmd = [sys.executable, str(ROOT / "tools/trial_log.py"), "record",
                   str(ROOT / "examples/synthetic-wet.json"), "--log", str(p)]
            first = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue(json.loads(first.stdout)["passed"])
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)
            result = subprocess.run([sys.executable, str(ROOT / "tools/trial_log.py"),
                                     "summary", str(p)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["physical_trials"], 0)


if __name__ == "__main__":
    unittest.main()
