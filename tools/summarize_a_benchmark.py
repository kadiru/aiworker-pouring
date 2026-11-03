"""Summarize the six actual reserved tests; never fills in missing outcomes."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'evidence/learning-A')
    parser.add_argument('--run',default='20261102T232206Z')
    args=parser.parse_args();root=args.evidence
    result=json.loads((root/'study-results.json').read_text())
    model=json.loads((root/'model-summary-v5.json').read_text())
    plan=json.loads((root/'frozen-plan.json').read_text())
    specs={s['name']:s for s in plan['test']}
    evaluated=result['policy_by_checkpoint'][model['checkpoint_sha256']]
    tests=[r for r in evaluated.get('test_policy',{}).get('rows',[]) if '/'+args.run+'/' in r['path']]
    assert len(tests)==6, f'Reserved benchmark incomplete or duplicated: {len(tests)} outcomes'
    assert len({r['name'] for r in tests})==6, 'Duplicate test cases'
    rows=[]
    for r in sorted(tests,key=lambda x:x['name']):
        name=r['name'].removeprefix('BC_');spec=specs[name]
        quantum=r['spill']['particle_quantum_ml']
        start_cup=round(spec['cup_initial_ml']/quantum)*quantum
        delivered=r['cup_ml']-start_cup
        if abs(delivered)<quantum/2:delivered=0.
        rows.append(dict(case=name,source_initial_ml=spec['initial_ml'],cup_initial_ml=spec['cup_initial_ml'],
            grasp_verified=r['grasp'],delivered_to_cup_ml=delivered,cup_final_ml=r['cup_ml'],
            cup_final_percent=100*r['cup_fraction'],bottle_remaining_ml=r['bottle_ml'],
            final_outside_ml=r['spill']['final_outside_ml'],ever_below_worktop_ml=r['spill']['ever_table_ml'],
            predicted_stop=r['stop'],expected_stop=r['expected_stop'],policy_completed=r['policy']['done'],
            operational_zero_spill_success=r['operational_zero_spill_success'],
            strict_zero_spill_success=r['strict_zero_spill_success'],gates='; '.join(r['gates']),
            native_trial_path=r['path']))
    summary=dict(status='complete',evidence_kind='native Isaac Sim learned-policy rollouts',run=args.run,model=model,
        cases=len(rows),verified_grasps=sum(r['grasp_verified'] for r in rows),
        transferred_at_least_1_ml=sum(r['delivered_to_cup_ml']>=1 for r in rows),
        autonomous_completions=sum(r['policy_completed'] for r in rows),
        operational_zero_spill_successes=sum(r['operational_zero_spill_success'] for r in rows),
        strict_zero_spill_successes=sum(r['strict_zero_spill_success'] for r in rows),
        final_outside_mean_ml=sum(r['final_outside_ml'] for r in rows)/6,
        final_outside_max_ml=max(r['final_outside_ml'] for r in rows),
        definitions=dict(operational_success='Verified sustained grip, policy completion, upright set-down, zero escaped/below-worktop particles, and cup >=90% brim OR source residual <=max(0.25 mL,1% initial source).',
            strict_success='Operational conditions plus the originally declared stop-reason classification gates.',
            volume_units='Nominal particle volume; not physical millilitre calibration.',
            horizons='8 s grasp deadline, 32 s overall simulation horizon; control 20 Hz.'),rows=rows)
    (root/'final-benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'final-benchmark.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('model','definitions','rows')},indent=2))


if __name__=='__main__':main()
