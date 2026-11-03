"""Audit the frozen visual-stop suite; preserve legacy inventory scoring."""
import argparse
import hashlib
import json
from pathlib import Path

from pour_stop_study import ROOT


def report(require_six=False):
    sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest()
    plan=json.loads((ROOT/'frozen-plan.json').read_text())
    groups={'validation':[],'test':[]};all_attempts=[]
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*/metrics.json')):
        spec=json.loads((p.parent/'parameters.json').read_text())
        if not spec.get('name','').startswith(('S_A_','replay_S_A_')):continue
        assert spec['spacing']==.0018 and spec['dt']==.0025
        m=json.loads(p.read_text());f=m['bottle_result'];policy=m['policy']
        samples=json.loads((p.parent/'samples.json').read_text())
        decisions=json.loads((p.parent/'policy-decisions.json').read_text())
        stop=next((r for r in samples if r['stop_command']),None)
        first=next((r for r in decisions if r.get('stop_latched')),None)
        quantum=spec['spacing']**3*1e6;baseline=round(spec.get('cup_initial_ml',0.)/quantum)*quantum
        source_empty=f['bottle_ml']<=max(.25,.01*spec['initial_ml'])
        row=dict(name=spec['name'],path=str(p.parent),checkpoint_sha256=policy['checkpoint_sha256'],
            source_initial_ml=spec['initial_ml'],cup_initial_ml=spec.get('cup_initial_ml',0.),
            grasp_verified=m['grasp_verified'],sustained_grasp_fraction=m['bilateral_fraction_during_pour'],
            delivered_to_cup_ml=max(0.,f['cup_ml']-baseline),cup_final_ml=f['cup_ml'],
            cup_surface_height_fraction=f['cup_fill_fraction'],cup_nominal_inventory_fraction=f['cup_inventory_fraction'],
            source_remaining_ml=f['bottle_ml'],source_empty=bool(source_empty),
            stop_reason=f['stop_reason'],expected_legacy_stop=m['expected_stop'],
            final_outside_ml=m['spill']['final_outside_ml'],ever_below_worktop_ml=m['spill']['ever_table_ml'],
            zero_spill=m['zero_spill'],policy_completed=policy['done'],
            upright_set_down=bool(abs(f['bottle_tilt_deg'])<8 and abs(f['bottle_base_height_m'])<=.005),
            surface_goal_zero_spill_success=m.get('surface_task_success_zero_spill',False),
            legacy_inventory_zero_spill_success=m.get('task_success_zero_spill',False),
            physical_surface_or_empty_condition=bool(f['cup_fill_fraction']>=.90 or source_empty),
            teacher_fallback=policy['teacher_fallback'],gates=m['gates'],sim_seconds=f['t'],wall_seconds=m['wall_seconds'],
            stop_observation=None if stop is None else {k:stop[k] for k in ('t','cup_fill_fraction','cup_inventory_fraction','bottle_ml','airborne_ml','spilled_ml')},
            stop_prediction=first,
            terminal_modes=list(dict.fromkeys(r['mode'] for r in decisions if r.get('stop_latched'))))
        all_attempts.append(row)
        if policy['checkpoint_sha256']==sha:groups[spec['study_split']].append(row)
    result=dict(checkpoint_sha256=sha,scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        volume_units='nominal particle volume; not calibrated mL',
        primary_definition='surface height >=90% rim OR source <=max(.25 nominal mL,1% initial), correct visual reason, verified sustained grasp, completed upright set-down/release, zero outside/below-worktop particles',
        legacy_definition='original nominal cup inventory >=90% or source empty, plus original physical gates and completion',
        goal_calibration_note='Declared before native tests; original inventory criterion retained separately. Fresh positions/fills/physics unchanged. Render budget increased to avoid disappearing liquid mesh.',
        all_attempts=all_attempts,groups={})
    for split,rows in groups.items():
        result['groups'][split]=dict(attempts=len(rows),verified_grasps=sum(r['grasp_verified'] for r in rows),
            liquid_transfers_at_least_1_ml=sum(r['delivered_to_cup_ml']>=1. for r in rows),
            policy_completions=sum(r['policy_completed'] for r in rows),zero_spill_runs=sum(r['zero_spill'] for r in rows),
            surface_goal_zero_spill_successes=sum(r['surface_goal_zero_spill_success'] for r in rows),
            legacy_inventory_zero_spill_successes=sum(r['legacy_inventory_zero_spill_success'] for r in rows),
            final_outside_mean_ml=sum(r['final_outside_ml'] for r in rows)/len(rows) if rows else None,rows=rows)
    tests=groups['test'];expected={s['name'] for s in plan['test']}
    complete=len(tests)==6 and {r['name'] for r in tests}==expected
    result['test_status']='complete' if complete else 'incomplete'
    if require_six:
        assert complete,f'Expected exactly six frozen final-checkpoint cases, got {len(tests)}'
        assert all(not r['teacher_fallback'] for r in tests)
    (ROOT/'stop-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(test_status=result['test_status'],checkpoint_sha256=sha,
        groups={k:{a:b for a,b in v.items() if a!='rows'} for k,v in result['groups'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-six',action='store_true');args=p.parse_args();report(args.require_six)
