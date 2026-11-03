#!/usr/bin/env python3
"""Offline pouring trial log. Python 3.9+, standard library, no robot commands."""
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROTOCOL = "pour-v1"
MODES = ("teleoperated", "scripted", "learned", "mixed")
KINDS = ("physical", "simulation", "synthetic")
PHASES = ("dry", "wet")
FLAGS = ("cycle_complete", "intervention", "collision", "object_drop", "estop",
         "spill_outside_tray")
INCIDENTS = ("collision", "object_drop", "estop", "spill_outside_tray")
MASSES = ("initial_water_g", "cup_before_g", "cup_after_g", "bottle_before_g",
          "bottle_after_g", "spill_before_g", "spill_after_g")
TEXT = ("trial_id", "series_id", "started_at", "robot_model", "setup_id",
        "software_ref", "mode_detail", "artifact_ref")
OPTIONAL_TEXT = ("policy_ref", "episode_ref")
GROUP = ("evidence_kind", "phase", "mode", "robot_model", "setup_id",
         "software_ref", "policy_ref", "series_id")
FIELDS = set(TEXT + OPTIONAL_TEXT + FLAGS + MASSES +
             ("protocol", "evidence_kind", "phase", "mode", "duration_s", "notes"))


def template(trial_id, series_id, kind, phase, mode):
    r = {key: None for key in FIELDS}
    r.update({key: "" for key in TEXT})
    r.update(protocol=PROTOCOL, trial_id=trial_id, series_id=series_id,
             started_at=datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
             evidence_kind=kind, phase=phase, mode=mode, notes="")
    return r


def validate(r):
    if not isinstance(r, dict) or set(r) != FIELDS:
        missing = FIELDS - set(r) if isinstance(r, dict) else FIELDS
        extra = set(r) - FIELDS if isinstance(r, dict) else set()
        raise ValueError("wrong fields; missing=%s extra=%s" % (sorted(missing), sorted(extra)))
    if r["protocol"] != PROTOCOL:
        raise ValueError("unsupported protocol")
    for key, choices in (("evidence_kind", KINDS), ("mode", MODES), ("phase", PHASES)):
        if r[key] not in choices:
            raise ValueError("%s must be one of %s" % (key, choices))
    for key in TEXT:
        if not isinstance(r[key], str) or not r[key].strip():
            raise ValueError(key + " must be a nonempty string")
    for key in OPTIONAL_TEXT:
        if r[key] is not None and (not isinstance(r[key], str) or not r[key].strip()):
            raise ValueError(key + " must be null or a nonempty string")
    if not isinstance(r["notes"], str):
        raise ValueError("notes must be a string")
    try:
        dt = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError("started_at must be an ISO timestamp") from e
    if dt.utcoffset() is None:
        raise ValueError("started_at must include a UTC offset")
    if r["mode"] == "learned" and r["policy_ref"] is None:
        raise ValueError("learned trials require policy_ref")
    for key in FLAGS:
        if type(r[key]) is not bool:
            raise ValueError(key + " must be explicitly true or false")
    for key in MASSES + ("duration_s",):
        x = r[key]
        if x is not None and (type(x) not in (int, float) or not math.isfinite(x) or x < 0):
            raise ValueError(key + " must be null or a finite nonnegative number")
    if r["phase"] == "dry" and any(r[k] is not None for k in MASSES):
        raise ValueError("dry trials use null mass fields; no liquid measurement claim")
    return r


def evaluate(r):
    validate(r)
    reasons = []
    if not r["cycle_complete"]:
        reasons.append("cycle_incomplete")
    if r["intervention"]:
        reasons.append("intervention")
    reasons.extend(k for k in INCIDENTS if r[k])
    if r["duration_s"] is None:
        reasons.append("duration_unmeasured")
    elif not 0 < r["duration_s"] <= 60:
        reasons.append("duration_outside_0_to_60_s")
    metrics = {}
    if r["phase"] == "wet":
        if any(r[k] is None for k in MASSES):
            reasons.append("mass_unmeasured")
        else:
            delivered = r["cup_after_g"] - r["cup_before_g"]
            lost = r["bottle_before_g"] - r["bottle_after_g"]
            spill = r["spill_after_g"] - r["spill_before_g"]
            balance = lost - delivered - spill
            metrics = dict(delivered_g=round(delivered, 6), bottle_loss_g=round(lost, 6),
                           spill_g=round(spill, 6), mass_balance_error_g=round(balance, 6))
            if not 149 <= r["initial_water_g"] <= 151:
                reasons.append("initial_water_outside_150_plus_minus_1_g")
            if not 45 <= delivered <= 55:
                reasons.append("delivered_outside_50_plus_minus_5_g")
            if not 0 <= spill <= 2:
                reasons.append("spill_outside_0_to_2_g")
            if lost < 0 or lost > r["initial_water_g"] or abs(balance) > 3:
                reasons.append("mass_accounting_invalid")
    return dict(passed=not reasons, reasons=reasons, metrics=metrics)


def decode(text):
    # JSON's permissive NaN handling and duplicate keys would hide input mistakes.
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("duplicate JSON key: " + k)
            obj[k] = v
        return obj
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=bad_constant)


def parse_log(content):
    records, ids = [], set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = validate(decode(line))
            if r["trial_id"] in ids:
                raise ValueError("duplicate trial_id: " + r["trial_id"])
        except (ValueError, TypeError) as e:
            raise ValueError("log line %s: %s" % (line_number, e)) from e
        records.append(r)
        ids.add(r["trial_id"])
    return records


def record(path, r):
    validate(r)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        content = f.read()
        records = parse_log(content)
        if any(old["trial_id"] == r["trial_id"] for old in records):
            raise ValueError("duplicate trial_id: " + r["trial_id"])
        same_series = [old for old in records if old["series_id"] == r["series_id"]]
        if any(tuple(old[k] for k in GROUP) != tuple(r[k] for k in GROUP) for old in same_series):
            raise ValueError("series_id already has different conditions; start a new series")
        if same_series and datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")) < max(
                datetime.fromisoformat(old["started_at"].replace("Z", "+00:00")) for old in same_series):
            raise ValueError("record series attempts in chronological order")
        f.seek(0, os.SEEK_END)
        if content and not content.endswith("\n"):
            f.write("\n")
        f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return evaluate(r)


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        validate(r)
        groups[tuple(r[k] for k in GROUP)].append(r)
    out = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")))
        results = [evaluate(r) for r in rows]
        last = rows[-5:]
        gate = None
        if key[0] == "physical" and key[1] == "wet" and len(last) == 5:
            gate = sum(evaluate(r)["passed"] for r in last) >= 4 and not any(
                r[k] for r in last for k in INCIDENTS)
        out.append(dict(conditions=dict(zip(GROUP, key)), attempts=len(rows),
                        passed=sum(x["passed"] for x in results),
                        last_five_ids=[r["trial_id"] for r in last],
                        meets_physical_five_trial_rule=gate,
                        trials=[dict(trial_id=r["trial_id"], **v) for r, v in zip(rows, results)]))
    return dict(protocol=PROTOCOL, physical_trials=sum(r["evidence_kind"] == "physical" for r in records),
                warning="Operator-entered evidence; this tool does not verify artifacts or robot execution.",
                groups=out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("template", help="Print an incomplete form; fill it before recording")
    p.add_argument("--trial-id", required=True)
    p.add_argument("--series-id", required=True)
    p.add_argument("--evidence-kind", choices=KINDS, default="synthetic")
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p = sub.add_parser("validate", help="Validate and evaluate one JSON trial")
    p.add_argument("input", type=Path)
    p = sub.add_parser("record", help="Append one JSON trial to a locked JSONL log")
    p.add_argument("input", type=Path)
    p.add_argument("--log", type=Path, required=True)
    p = sub.add_parser("summary", help="Summarize JSONL trials without mixing conditions")
    p.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            output = template(args.trial_id, args.series_id, args.evidence_kind, args.phase, args.mode)
        elif args.command == "summary":
            with args.log.open(encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                output = summarize(parse_log(f.read()))
        else:
            r = validate(decode(args.input.read_text(encoding="utf-8")))
            output = record(args.log, r) if args.command == "record" else evaluate(r)
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError) as e:
        print("error: " + str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
