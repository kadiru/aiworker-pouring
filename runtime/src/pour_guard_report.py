"""Fresh guarded-policy benchmark, all development failures retained."""
import argparse,hashlib,json
from pathlib import Path
from pour_guard_study import ROOT


def report(require_six=False):
    sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest();plan=json.loads((ROOT/'frozen-plan.json').read_text())
    groups={s:[] for s in ('regression','validation','test')};attempts=[]
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*G_A_*/metrics.json')):
        spec=json.loads((p.parent/'parameters.json').read_text());m=json.loads(p.read_text());f=m['bottle_result'];policy=m['policy']
        assert spec['spacing']==.0018 and spec['dt']==.0025
        rows=json.loads((p.parent/'samples.json').read_text());d=json.loads((p.parent/'policy-decisions.json').read_text())
        stop=next((x for x in rows if x['stop_command']),None);quantum=spec['spacing']**3*1e6
        baseline=round(spec['cup_initial_ml']/quantum)*quantum
        row=dict(name=spec['name'],path=str(p.parent),checkpoint_sha256=policy['checkpoint_sha256'],
            source_initial_ml=spec['initial_ml'],cup_initial_ml=spec['cup_initial_ml'],grasp_verified=m['grasp_verified'],
            sustained_grasp_fraction=m['bilateral_fraction_during_pour'],grasp_attempts=max(x.get('grasp_attempt',0) for x in d),
            visual_grasp_confirmed=any(x.get('visual_grasp_confirmed',False) for x in d),
            delivered_to_cup_ml=max(0.,f['cup_ml']-baseline),cup_final_ml=f['cup_ml'],
            cup_surface_height_fraction=f['cup_fill_fraction'],cup_nominal_inventory_fraction=f['cup_inventory_fraction'],
            source_remaining_ml=f['bottle_ml'],stop_reason=f['stop_reason'],final_outside_ml=m['spill']['final_outside_ml'],
            ever_below_worktop_ml=m['spill']['ever_table_ml'],zero_spill=m['zero_spill'],policy_completed=policy['done'],
            surface_goal_zero_spill_success=m['surface_task_success_zero_spill'],legacy_inventory_zero_spill_success=m['task_success_zero_spill'],
            teacher_fallback=policy['teacher_fallback'],gates=m['gates'],sim_seconds=f['t'],wall_seconds=m['wall_seconds'],
            pause_count=max(x.get('pause_count',0) for x in d),
            stop_prediction=next((x for x in d if x['stop_latched']),None),
            stop_observation=None if stop is None else {k:stop[k] for k in ('t','cup_fill_fraction','cup_inventory_fraction','bottle_ml','spilled_ml','airborne_ml')},
            terminal_modes=list(dict.fromkeys(x['mode'] for x in d if x['stop_latched'])))
        row['completed_return_release_sequence']=bool(policy['done'] and f['stop_reason'] in ('cup_near_full','bottle_empty') and
            all(mode in row['terminal_modes'] for mode in ('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose')))
        attempts.append(row)
        if policy['checkpoint_sha256']==sha:groups[spec['study_split']].append(row)
    result=dict(checkpoint_sha256=sha,scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        volume_units='nominal particle count volume; not calibrated mL',
        primary_definition='verified sustained grasp, height>=90%rim OR source<=max(.25 nominal mL,1% initially), correct reason, completed upright set-down/release, zero measured outside/below-worktop particles',
        legacy_definition='Original inventory>=90% plus original physical/completion gates; retained unchanged',
        test_policy='New seed20261122; one fixed checkpoint, no expert fallback or tuning during six reserved tests',all_attempts=attempts,groups={})
    for split,rs in groups.items():
        result['groups'][split]=dict(attempts=len(rs),verified_grasps=sum(x['grasp_verified'] for x in rs),
            visual_confirmations=sum(x['visual_grasp_confirmed'] for x in rs),
            liquid_transfers_at_least_1_ml=sum(x['delivered_to_cup_ml']>=1. for x in rs),
            policy_completions=sum(x['policy_completed'] for x in rs),zero_spill_runs=sum(x['zero_spill'] for x in rs),
            return_release_sequences=sum(x['completed_return_release_sequence'] for x in rs),
            surface_goal_zero_spill_successes=sum(x['surface_goal_zero_spill_success'] for x in rs),
            legacy_inventory_zero_spill_successes=sum(x['legacy_inventory_zero_spill_success'] for x in rs),
            final_outside_mean_ml=sum(x['final_outside_ml'] for x in rs)/len(rs) if rs else None,rows=rs)
    tests=groups['test'];complete=len(tests)==6 and {x['name'] for x in tests}=={x['name'] for x in plan['test']}
    result['test_status']='complete' if complete else 'incomplete'
    if require_six:
        assert complete,f'Expected exactly six reserved cases, got {len(tests)}'
        assert not any(x['teacher_fallback'] for x in tests)
    (ROOT/'guard-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(test_status=result['test_status'],checkpoint_sha256=sha,groups={s:{k:v for k,v in g.items() if k!='rows'} for s,g in result['groups'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-six',action='store_true');report(p.parse_args().require_six)
