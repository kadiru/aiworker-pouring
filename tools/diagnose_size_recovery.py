"""Read-only outcome diagnosis for all six consumed-layout size-stage regressions."""
import hashlib
import json
import sys
from pathlib import Path

sys.path[:0] = ['/workspace/src', '/workspace']
from pour_size_study import ROOT, OUTPUTS
from pour_robust_report import completed_row
from pour_wide_report import strict_success, reasons


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(folder, checkpoint):
    row = completed_row(folder, checkpoint)
    samples = read(folder / 'samples.json')
    decisions = read(folder / 'policy-decisions.json')
    stop = next((d for d in decisions if d.get('stop_latched')), None)
    loss = next((s for s in samples if s['ever_table_ml'] > 0), None)
    source_zero = next((s for s in samples if s['bottle_ml'] == 0), None)
    # Decision time precedes its recorded next-state sample by one control step.
    closest = None if stop is None else min(samples, key=lambda s: abs(s['t'] - stop['t']))
    terminal = read(folder / 'metrics.json')['bottle_result']
    assert terminal == samples[-1]
    stages = {}
    for d in decisions:
        if d.get('stop_latched'):
            stages.setdefault(d['mode'], d['t'])
    settling = [d for d in decisions if d.get('fill_mode') == 'settle' and not d.get('stop_latched')]
    keep = ('t', 'mode', 'fill_mode', 'pause_count', 'cup_predicted_fraction',
            'surface_geometry_fraction', 'settled_prediction', 'empty_probability',
            'source_present_probability', 'visual_flow_fraction_s',
            'estimated_tilt_from_joints', 'source_watchdog', 'stop_latched')
    return dict(
        name=row['name'], strict_pass=strict_success(row), failure_reasons=reasons(row),
        expected_stop=row['expected_stop'], actual_stop=row['stop_reason'],
        final_height=terminal['cup_fill_fraction'], source_remaining_nominal_ml=terminal['bottle_ml'],
        final_outside_nominal_ml=row['final_outside_ml'], ever_table_nominal_ml=row['ever_below_worktop_ml'],
        return_release_complete=row['completed_return_release_sequence'],
        policy_completed=row['policy_completed'], grasp_contact_fraction=row['sustained_grasp_fraction'],
        first_source_zero_time=None if source_zero is None else source_zero['t'],
        first_worktop_loss=loss, stop_decision=stop, sample_nearest_stop=closest,
        stop_minus_first_loss_s=None if stop is None or loss is None else stop['t'] - loss['t'],
        terminal_sample=terminal, return_stage_first_times=stages,
        last_settle_decisions=[{k: d.get(k) for k in keep} for d in settling[-10:]],
        hashes={n: sha(folder / n) for n in ('parameters.json', 'samples.json', 'policy-decisions.json', 'metrics.json')},
    )


def main():
    candidate = read(ROOT / 'candidate.json')
    rows = [diagnose(p.parent, candidate['checkpoint_sha256'])
            for p in sorted(OUTPUTS.glob('*/*SZ_A_recovery_*/metrics.json'))]
    assert len(rows) <= 6 and len({r['name'] for r in rows}) == len(rows)
    result = dict(
        scope='Consumed-layout development diagnosis. Measured particles/poses are scoring observations only, never policy inputs. Timing does not establish the physical mechanism of loss.',
        checkpoint_sha256=candidate['checkpoint_sha256'], cases=rows,
        completed=len(rows), strict_passes=sum(r['strict_pass'] for r in rows),
        zero_measured_loss=sum(r['final_outside_nominal_ml'] == r['ever_table_nominal_ml'] == 0 for r in rows),
        return_release_sequences=sum(r['return_release_complete'] for r in rows),
        helper_sha256=sha(Path(__file__)),
    )
    (ROOT / 'recovery-diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'scope')}))


if __name__ == '__main__':
    main()
