"""Summarize every attempt in frozen A study, including failed learned rollouts."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/learning/A-20261102-v1')


def aggregate(rows):
    escaped=[r['spill'].get('final_outside_ml',0) for r in rows]
    return dict(attempts=len(rows),physical_success=sum(r['physical_success'] for r in rows),
      operational_zero_spill_success=sum(r.get('operational_zero_spill_success',False) for r in rows),
      strict_zero_spill_success=sum(r['strict_zero_spill_success'] for r in rows),
      zero_spill=sum(r['zero_spill'] for r in rows),grasp=sum(r['grasp'] for r in rows),
      stop_label_correct=sum(r['stop_label_correct'] for r in rows),
      stop_condition_achieved=sum(r['stop_condition_achieved'] for r in rows),
      final_outside_mean_ml=float(np.mean(escaped)) if escaped else None,
      final_outside_max_ml=max(escaped) if escaped else None,rows=rows)


def report():
    groups={k:[] for k in ('train','corrective_approach','validation_reference','validation_policy','test_policy')}
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*/metrics.json')):
        spec=json.loads((p.parent/'parameters.json').read_text())
        if not (spec.get('name','').startswith(('A_train_','A_validation_','BC_A_')) or
                (spec.get('policy_evaluation') and spec.get('study_split') in ('validation','test'))): continue
        m=json.loads(p.read_text());f=m['bottle_result']
        split=spec['study_split']
        group=('corrective_approach' if spec.get('corrective_prefix') else split+'_policy' if spec.get('policy_evaluation') else
               'validation_reference' if split=='validation' else 'train')
        row=dict(name=spec['name'],path=str(p.parent),physical_success=m['eligible'],
             strict_zero_spill_success=m.get('task_success_zero_spill',False),
             zero_spill=m.get('zero_spill',False),grasp=m['grasp_verified'],
             stop_label_correct=f['stop_reason']==m['expected_stop'],
             stop_condition_achieved=(f['stop_reason']==m['expected_stop'] and
                 (f['cup_inventory_fraction']>=.90 if m['expected_stop']=='cup_near_full' else
                  f['bottle_ml']<=max(.25,.01*spec['initial_ml'])) and
                 abs(f['bottle_tilt_deg'])<8),
             stop=f['stop_reason'],expected_stop=m['expected_stop'],
             cup_ml=f['cup_ml'],cup_fraction=f['cup_inventory_fraction'],
             bottle_ml=f['bottle_ml'],spill=m.get('spill',{}),gates=m['gates'],
             policy=m.get('policy'),sim_seconds=f['t'],wall_seconds=m['wall_seconds'])
        row.update(task_scope=m.get('task_scope','full_pour'),prefix_success=m.get('prefix_success'))
        volume_stop=f['cup_inventory_fraction']>=.90 or f['bottle_ml']<=max(.25,.01*spec['initial_ml'])
        finished=m.get('policy',{}).get('done',False) if spec.get('policy_evaluation') else True
        row['volume_stop_condition']=bool(volume_stop)
        row['operational_zero_spill_success']=bool(not spec.get('corrective_prefix') and finished and
            m['grasp_verified'] and m['bilateral_fraction_during_pour']>=.8 and m.get('zero_spill') and
            abs(f['bottle_tilt_deg'])<8 and abs(f['bottle_base_height_m'])<=.005 and volume_stop)
        groups[group].append(row)
    result={}
    for group,rows in groups.items():
        result[group]=aggregate(rows)
    models={}
    for group in ('validation_policy','test_policy'):
        rows=groups[group]
        for sha in {r['policy']['checkpoint_sha256'] for r in rows}:
            models.setdefault(sha,{})[group]=aggregate([r for r in rows if r['policy']['checkpoint_sha256']==sha])
    result['policy_by_checkpoint']=models
    (ROOT/'study-results.json').write_text(json.dumps(result,indent=2)+'\n')
    compact={k:{a:b for a,b in v.items() if a!='rows'} for k,v in result.items() if k!='policy_by_checkpoint'}
    compact['policy_by_checkpoint']={sha:{g:{k:v for k,v in r.items() if k!='rows'} for g,r in groups.items()} for sha,groups in models.items()}
    print(json.dumps(compact,indent=2))


if __name__=='__main__': report()
