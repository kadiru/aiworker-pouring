"""All thirty frozen outcomes, including every failed or incomplete attempt."""
import argparse,csv,hashlib,json
from pathlib import Path
from pour_robust_study import ROOT,PREFIX,CHECKPOINT_SHA,plan


def completed_row(folder, checkpoint_sha=CHECKPOINT_SHA):
    s=json.loads((folder/'parameters.json').read_text());m=json.loads((folder/'metrics.json').read_text())
    f=m['bottle_result'];p=m['policy'];d=json.loads((folder/'policy-decisions.json').read_text());samples=json.loads((folder/'samples.json').read_text())
    assert p['checkpoint_sha256']==checkpoint_sha and not p['teacher_fallback']
    assert s['spacing']==.0018 and s['dt']==.0025 and s['vessel_appearance']=='A'
    q=s['spacing']**3*1e6;baseline=round(s['cup_initial_ml']/q)*q
    modes=list(dict.fromkeys(x['mode'] for x in d if x['stop_latched']))
    sequence=p['done'] and all(x in modes for x in ('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose'))
    return dict(name=s['name'],status='complete',path=str(folder),expected_stop=s['expected_stop'],
        checkpoint_sha256=p['checkpoint_sha256'],source_initial_ml=s['initial_ml'],cup_initial_ml=s['cup_initial_ml'],
        cup_world=s['cup_world'],source_xy=s['source_xy'],grasp_verified=m['grasp_verified'],
        visual_grasp_confirmed=any(x.get('visual_grasp_confirmed',False) for x in d),
        grasp_attempts=max(x.get('grasp_attempt',0) for x in d),sustained_grasp_fraction=m['bilateral_fraction_during_pour'],
        delivered_to_cup_ml=max(0.,f['cup_ml']-baseline),cup_final_ml=f['cup_ml'],
        cup_surface_height_fraction=f['cup_fill_fraction'],cup_nominal_inventory_fraction=f['cup_inventory_fraction'],
        source_remaining_ml=f['bottle_ml'],stop_reason=f['stop_reason'],final_outside_ml=m['spill']['final_outside_ml'],
        ever_below_worktop_ml=m['spill']['ever_table_ml'],peak_nonairborne_outside_ml=max(x['spilled_ml'] for x in samples),
        zero_spill=m['zero_spill'],policy_completed=p['done'],completed_return_release_sequence=bool(sequence),
        surface_goal_zero_spill_success=m['surface_task_success_zero_spill'],legacy_inventory_zero_spill_success=m['task_success_zero_spill'],
        teacher_fallback=p['teacher_fallback'],gates=m['gates'],sim_seconds=f['t'],wall_seconds=m['wall_seconds'],
        stop_prediction=next((x for x in d if x['stop_latched']),None))


def report(require_thirty=False):
    selection=json.loads((ROOT/'final-selection.json').read_text());frozen=plan();base=Path('/workspace/outputs/fountain')
    assert hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest()==CHECKPOINT_SHA
    assert hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest()==selection['scene_plan_sha256']
    folders={}
    for p in base.glob(f'*/*{PREFIX}*/parameters.json'):
        s=json.loads(p.read_text());name=s['name'];assert name not in folders,'Duplicate reserved attempt: '+name;folders[name]=p.parent
    rows=[]
    for s in frozen['test']:
        f=folders.get(s['name'])
        if f is None:rows.append(dict(name=s['name'],status='pending',expected_stop=s['expected_stop']));continue
        actual=json.loads((f/'parameters.json').read_text())
        for key in s:assert actual[key]==s[key],(s['name'],key)
        if (f/'metrics.json').exists():rows.append(completed_row(f))
        else:rows.append(dict(name=s['name'],status='incomplete',path=str(f),expected_stop=s['expected_stop']))
    done=[r for r in rows if r['status']=='complete'];counts=dict(planned=30,completed=len(done),pending=sum(x['status']=='pending' for x in rows),incomplete=sum(x['status']=='incomplete' for x in rows))
    for out,key in (('verified_grasps','grasp_verified'),('visual_confirmations','visual_grasp_confirmed'),('return_release_sequences','completed_return_release_sequence'),
        ('primary_successes','surface_goal_zero_spill_success'),('legacy_successes','legacy_inventory_zero_spill_success'),('zero_spill_runs','zero_spill')):
        counts[out]=sum(x[key] for x in done)
    counts['liquid_transfers_at_least_1_ml']=sum(x['delivered_to_cup_ml']>=1. for x in done)
    counts['final_outside_mean_ml']=sum(x['final_outside_ml'] for x in done)/len(done) if done else None
    counts['final_outside_max_ml']=max((x['final_outside_ml'] for x in done),default=None)
    groups={}
    for reason in ('cup_near_full','bottle_empty'):
        group=[x for x in done if x['expected_stop']==reason]
        groups[reason]=dict(planned=15,completed=len(group),successes=sum(x['surface_goal_zero_spill_success'] for x in group),
            stop_reason_errors=sum(x['stop_reason']!=reason for x in group))
    result=dict(status='complete' if len(done)==30 else 'incomplete',counts=counts,groups=groups,cases=rows,
        checkpoint_sha256=CHECKPOINT_SHA,scene_plan_sha256=selection['scene_plan_sha256'],selection=selection,
        primary_definition=selection['primary_definition'],legacy_definition=selection['legacy_definition'],volume_units=selection['volume_units'],
        note='All30 remain in denominator; zero spill from no pour or underfill is not a task success. No fitting or parameter changes during evaluation.')
    if require_thirty:assert len(done)==30,'All thirty complete native outcomes are required'
    (ROOT/'robust-results.json').write_text(json.dumps(result,indent=2)+'\n')
    if len(done)==30:
        with (ROOT/'final-benchmark.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(done[0]),lineterminator='\n');w.writeheader()
            w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in x.items()} for x in done)
    print(json.dumps(dict(status=result['status'],counts=counts,groups=groups),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-thirty',action='store_true');report(p.parse_args().require_thirty)
