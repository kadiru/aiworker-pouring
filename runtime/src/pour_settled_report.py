"""All corrective attempts and the thirty new frozen outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from pour_robust_report import completed_row
from pour_settled_study import ROOT, PREFIX, plan


def report(require_thirty=False):
    sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest();p=plan();attempts=[];folders={};incomplete=[]
    for f in sorted(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json')):
        spec=json.loads(f.read_text());folder=f.parent
        assert spec['name'] not in folders, 'Duplicate reserved/attempt name'
        folders[spec['name']]=folder
        if (folder/'metrics.json').exists():
            m=json.loads((folder/'metrics.json').read_text())
            row=completed_row(folder,m['policy']['checkpoint_sha256'])
            row.update(split=spec['study_split'],development_case=spec.get('development_case'),development_round=spec.get('development_round'))
            attempts.append(row)
        else:
            incomplete.append(dict(name=spec['name'],path=str(folder),split=spec['study_split'],status='incomplete',
                partial_recording=str(folder/'episode.partial.hdf5'),note='Interrupted attempt; spill and task outcome are not imputed.'))
    rows=[]
    for spec in p['test']:
        folder=folders.get(spec['name'])
        if folder is None:rows.append(dict(name=spec['name'],status='pending',expected_stop=spec['expected_stop']));continue
        actual=json.loads((folder/'parameters.json').read_text())
        for k in spec:assert spec[k]==actual[k],(spec['name'],k)
        row=next((x for x in attempts if x['name']==spec['name']),None)
        if row is None:rows.append(dict(name=spec['name'],status='incomplete',expected_stop=spec['expected_stop'],path=str(folder)))
        else:
            assert row['checkpoint_sha256']==sha
            rows.append(row)
    done=[x for x in rows if x['status']=='complete']
    counts=dict(planned=30,completed=len(done),pending=sum(x['status']=='pending' for x in rows),incomplete=sum(x['status']=='incomplete' for x in rows))
    for out,key in (('verified_grasps','grasp_verified'),('visual_confirmations','visual_grasp_confirmed'),('return_release_sequences','completed_return_release_sequence'),
            ('primary_successes','surface_goal_zero_spill_success'),('legacy_successes','legacy_inventory_zero_spill_success'),('zero_spill_runs','zero_spill')):
        counts[out]=sum(x[key] for x in done)
    counts['liquid_transfers_at_least_1_ml']=sum(x['delivered_to_cup_ml']>=1 for x in done)
    counts['final_outside_mean_ml']=sum(x['final_outside_ml'] for x in done)/len(done) if done else None
    counts['final_outside_max_ml']=max((x['final_outside_ml'] for x in done),default=None)
    groups={}
    for reason in ('cup_near_full','bottle_empty'):
        g=[x for x in done if x['expected_stop']==reason]
        groups[reason]=dict(planned=15,completed=len(g),successes=sum(x['surface_goal_zero_spill_success'] for x in g),stop_reason_errors=sum(x['stop_reason']!=reason for x in g))
    development=[x for x in attempts if x['split']=='regression' and x['checkpoint_sha256']==sha]
    result=dict(status='complete' if len(done)==30 else 'incomplete',counts=counts,groups=groups,cases=rows,
        checkpoint_sha256=sha,scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        development=development,all_attempts=attempts,incomplete_attempts=incomplete,
        primary_definition='Sustained verified grasp, final rim height>=90% or source<=max(.25 nominal mL,1% initial), correct reason, completed return/release, zero final outside and ever-below-worktop particles',
        legacy_definition='Original >=90% nominal brim inventory score retained separately',volume_units='Nominal particle-count mL, not calibrated physical volume')
    if (ROOT/'final-selection.json').exists():result['selection']=json.loads((ROOT/'final-selection.json').read_text())
    if require_thirty:assert len(done)==30
    (ROOT/'settled-results.json').write_text(json.dumps(result,indent=2)+'\n')
    if len(done)==30:
        with (ROOT/'final-benchmark.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(done[0]),lineterminator='\n');w.writeheader()
            w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in x.items()} for x in done)
    print(json.dumps(dict(status=result['status'],counts=counts,development=[dict(name=x['name'],success=x['surface_goal_zero_spill_success'],height=x['cup_surface_height_fraction'],source=x['source_remaining_ml'],spill=x['final_outside_ml'],reason=x['stop_reason']) for x in development]),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-thirty',action='store_true');report(p.parse_args().require_thirty)
