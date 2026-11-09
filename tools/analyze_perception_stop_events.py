"""Describe recorded stop/spill timing from audited local sidecars, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / 'evidence/perception-A'
DATA = PROJECT / 'data/material-A-perception'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=('development', 'test'), required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    stem = f'{args.part}-r{args.round}'
    outcomes_path, audit_path = [EVIDENCE / (stem + suffix) for suffix in ('-outcomes.json', '-audit.json')]
    output = EVIDENCE / (stem + '-stop-events.json')
    assert not output.exists(), 'Preserve existing diagnostic'
    outcomes, audit = read(outcomes_path), read(audit_path)
    assert outcomes['status'] == 'complete' and not audit['started_incomplete']
    assert outcomes['audit_sha256'] == sha(audit_path)
    indexed = {r['name']: r for r in audit['records']}
    rows = []
    for result in outcomes['cases']:
        record = indexed[result['name']]
        assert record['sha256'] == result['source_sha256']
        folder = DATA / Path(record['path']).parent.relative_to('/workspace/outputs/fountain')
        for name, digest in record['sidecars'].items():
            assert sha(folder / name) == digest
        samples, decisions, spec = [read(folder / name) for name in ('samples.json', 'policy-decisions.json', 'parameters.json')]
        assert len(samples) == len(decisions)
        # Decisions describe the action at t; samples describe its next state.
        period = spec['control_period']
        assert all(abs(s['t'] - d['t'] - period) < 1e-6 for s, d in zip(samples, decisions))
        first = lambda predicate: next((i for i, s in enumerate(samples) if predicate(s)), None)
        stop = first(lambda s: s['stop_command'])
        loss = first(lambda s: s['ever_table_ml'] > 0)
        empty = first(lambda s: s['bottle_ml'] == 0)
        final = samples[-1]
        assert abs(final['cup_fill_fraction'] - result['cup_surface_height_fraction']) < 1e-8
        stop_time = samples[stop]['stop_time'] if stop is not None else None
        terminal = []
        for i in range(1, len(samples)):
            a, b = samples[i-1:i+1]
            if all(d['mode'] == 'Holding completed pose' for d in decisions[i-1:i+1]):
                if abs(b['cup_ml'] - a['cup_ml']) < 1e-9 and abs(b['bottle_ml'] - a['bottle_ml']) < 1e-9:
                    terminal.append(dict(start_s=a['t'], end_s=b['t'], height_change=b['cup_fill_fraction']-a['cup_fill_fraction']))
        row = dict(name=result['name'], strict_pass=result['strict_pass'], failure_reasons=result['failure_reasons'],
            source_sha256=record['sha256'], sidecars_sha256=record['sidecars'], decision_to_sample_delay_s=period,
            stop_reason=result['stop_reason'], stop_command_s=stop_time,
            first_zero_source_s=None if empty is None else samples[empty]['t'],
            first_near_table_loss_s=None if loss is None else samples[loss]['t'],
            near_table_loss_before_stop_s=None if loss is None or stop_time is None else stop_time-samples[loss]['t'],
            final_near_table_lost_particles=round(final['ever_table_ml']/(spec['spacing']**3*1e6)),
            final_near_table_loss_nominal_ml=final['ever_table_ml'], final_outside_nominal_ml=final['spilled_ml'],
            peak_transient_outside_nominal_ml=max(s['spilled_ml'] for s in samples),
            maximum_recorded_height=max(s['cup_fill_fraction'] for s in samples),
            first_stop_sample_height=None if stop is None else samples[stop]['cup_fill_fraction'],
            final_height=final['cup_fill_fraction'], final_source_nominal_ml=final['bottle_ml'],
            largest_terminal_constant_inventory_drop=min(terminal, key=lambda r: r['height_change'], default=None))
        rows.append(row)
    report = dict(cases=rows, checkpoint_sha256=outcomes['checkpoint_sha256'],
        outcomes_sha256=sha(outcomes_path), audit_sha256=sha(audit_path), helper_sha256=sha(Path(__file__)),
        scope='Read-only scalar event timing at 20 Hz. Near-table loss is a position-based scoring measurement, not a contact-force measurement. Transient outside classification may include particles that return. Height is the recorded instantaneous metric, capped by its existing definition. No particle escape mechanism or solver cause is inferred. No fitting, scoring or controller changes; original final-sample outcomes remain unchanged.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('name', 'strict_pass', 'final_near_table_lost_particles', 'near_table_loss_before_stop_s', 'first_stop_sample_height', 'final_height')} for r in rows]))


if __name__ == '__main__':
    main()
