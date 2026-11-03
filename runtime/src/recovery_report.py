"""Report hybrid corrections separately from new autonomous policy rollouts."""
import argparse
import hashlib
import json
from pathlib import Path


def report(root,outputs,require_six=False):
    checkpoint=root/'best.pt'
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None
    groups={'recovery_train':[],'recovery_validation':[],'autonomous_validation':[],'autonomous_test':[]}
    for path in sorted(outputs.glob('*/*/metrics.json')):
        spec=json.loads((path.parent/'parameters.json').read_text())
        name=spec.get('name','');hybrid=spec.get('expert_recovery',False)
        if not (name.startswith('R_A_') and hybrid or name.startswith('BC6_R_A_')):continue
        assert spec['spacing']==.0018 and spec['dt']==.0025,'Unexpected liquid preset in completed study'
        m=json.loads(path.read_text());f=m['bottle_result'];policy=m.get('policy',{})
        if not hybrid and sha and policy.get('checkpoint_sha256')!=sha:continue
        split=spec['study_split']
        key=('recovery_' if hybrid else 'autonomous_')+split
        if key not in groups:continue
        quantum=spec['spacing']**3*1e6
        baseline=round(spec.get('cup_initial_ml',0.)/quantum)*quantum
        delivered=f['cup_ml']-baseline
        if abs(delivered)<quantum/2:delivered=0.
        physical_stop=(f['cup_inventory_fraction']>=.90 or f['bottle_ml']<=max(.25,.01*spec['initial_ml']))
        operational=bool(not hybrid and policy.get('done') and m['grasp_verified'] and
            m['bilateral_fraction_during_pour']>=.8 and m['zero_spill'] and
            abs(f['bottle_tilt_deg'])<8 and abs(f['bottle_base_height_m'])<=.005 and physical_stop)
        groups[key].append(dict(name=name,path=str(path.parent),
            evidence_kind='policy_prefix_expert_recovery' if hybrid else 'learned_visual_policy',
            source_initial_ml=spec['initial_ml'],cup_initial_ml=spec.get('cup_initial_ml',0.),
            grasp_verified=m['grasp_verified'],expert_started=m.get('recovery',{}).get('started',False),
            expert_recovery_zero_spill=m.get('recovery_success_zero_spill',False),
            delivered_to_cup_ml=delivered,cup_final_ml=f['cup_ml'],cup_fraction=f['cup_inventory_fraction'],
            bottle_remaining_ml=f['bottle_ml'],stop_reason=f['stop_reason'],expected_stop=m['expected_stop'],
            final_outside_ml=m['spill']['final_outside_ml'],ever_below_worktop_ml=m['spill']['ever_table_ml'],
            policy_completed=policy.get('done',False),operational_zero_spill_success=operational,
            physical_checks_pass=m['eligible'],volume_stop_condition_achieved=bool(physical_stop),
            stop_label_correct=f['stop_reason']==m['expected_stop'],
            strict_zero_spill_success=m['task_success_zero_spill'],
            checkpoint_sha256=policy.get('checkpoint_sha256'),gates=m['gates'],sim_seconds=f['t']))
        groups[key][-1]['teacher_fallback']=policy.get('teacher_fallback',False)
    result=dict(checkpoint_sha256=sha,volume_units='nominal particle volume, not physical calibration',
        definitions=dict(hybrid='Learned prefix + privileged scripted recovery; never autonomous success',
            operational='Learned completion, verified sustained grasp, upright set-down, zero outside/below-worktop particles; cup >=90% brim OR source <=max(.25mL,1% initial source)',
            strict='Original declared gates including stop-reason classification'),groups={})
    for key,rows in groups.items():
        result['groups'][key]=dict(attempts=len(rows),verified_grasps=sum(r['grasp_verified'] for r in rows),
            successful_expert_recoveries=sum(r['expert_recovery_zero_spill'] for r in rows),
            liquid_transfers_at_least_1_ml=sum(r['delivered_to_cup_ml']>=1. for r in rows),
            autonomous_completions=sum(r['policy_completed'] for r in rows),
            operational_zero_spill_successes=sum(r['operational_zero_spill_success'] for r in rows),
            strict_zero_spill_successes=sum(r['strict_zero_spill_success'] for r in rows),rows=rows)
    tests=groups['autonomous_test']
    expected={'BC6_'+s['name'] for s in json.loads((root/'frozen-plan.json').read_text())['test']}
    complete=len(tests)==6 and {r['name'] for r in tests}==expected
    result['test_status']='complete' if complete else 'incomplete'
    if require_six:
        assert complete,f'Expected six frozen final-checkpoint tests; found {len(tests)}'
        assert not any(r['teacher_fallback'] for r in tests),'Expert fallback in autonomous test'
    (root/'recovery-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(test_status=result['test_status'],checkpoint_sha256=sha,
        groups={k:{a:b for a,b in v.items() if a!='rows'} for k,v in result['groups'].items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/workspace/learning/A-recovery-20261102-v1'))
    parser.add_argument('--outputs',type=Path,default=Path('/workspace/outputs/fountain'))
    parser.add_argument('--require-six',action='store_true');args=parser.parse_args()
    report(args.root,args.outputs,args.require_six)
