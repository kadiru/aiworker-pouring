"""Retain every wide-study attempt and independently score all reserved tests."""
import argparse,csv,hashlib,json
from pathlib import Path
from pour_robust_report import completed_row
from pour_workspace_report import reference_row
from pour_wide_study import ROOT,OUTPUTS,PREFIX,REFERENCE_PREFIX,plan


def strict_success(r):
    goal=(r['cup_surface_height_fraction']>=.9 if r['expected_stop']=='cup_near_full' else
          r['source_remaining_ml']<=max(.25,.01*r['source_initial_ml']))
    return bool(r['grasp_verified'] and r['visual_grasp_confirmed'] and r['sustained_grasp_fraction']>=.8
        and r['delivered_to_cup_ml']>=1 and r['policy_completed'] and r['completed_return_release_sequence']
        and r['stop_reason']==r['expected_stop'] and not r['gates'] and goal
        and r['final_outside_ml']==r['ever_below_worktop_ml']==0 and not r['teacher_fallback'])


def reasons(r):
    result=[]
    if not r['grasp_verified'] or not r['visual_grasp_confirmed']:result.append('grasp_unconfirmed')
    if r['sustained_grasp_fraction']<.8:result.append('grip_not_sustained')
    if r['delivered_to_cup_ml']<1:result.append('insufficient_transfer')
    if r['stop_reason']!=r['expected_stop']:result.append(r['stop_reason'] or 'no_stop')
    if not r['completed_return_release_sequence']:result.append('return_incomplete')
    if not r['policy_completed']:result.append('policy_incomplete')
    if r['teacher_fallback']:result.append('teacher_fallback')
    if r['expected_stop']=='cup_near_full' and r['cup_surface_height_fraction']<.9:result.append('underfilled')
    if r['expected_stop']=='bottle_empty' and r['source_remaining_ml']>max(.25,.01*r['source_initial_ml']):result.append('source_not_empty')
    if r['final_outside_ml']>0 or r['ever_below_worktop_ml']>0:result.append('measured_spill')
    result.extend(r['gates']);return list(dict.fromkeys(result))


def report(require_thirty=False):
    p=plan();sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest() if (ROOT/'best.pt').exists() else None
    folders={};attempts=[];incomplete=[]
    for prefix in (PREFIX,REFERENCE_PREFIX+'demo_'):
        for path in sorted(OUTPUTS.glob(f'*/*{prefix}*/parameters.json')):
            s=json.loads(path.read_text())
            if s.get('protocol')!=17:continue
            assert s['name'] not in folders,'Duplicate attempt '+s['name'];folders[s['name']]=path.parent
            if not (path.parent/'metrics.json').exists():
                incomplete.append(dict(name=s['name'],path=str(path.parent),status='incomplete',split=s['study_split']));continue
            m=json.loads((path.parent/'metrics.json').read_text())
            r=completed_row(path.parent,m['policy']['checkpoint_sha256']) if s['policy_evaluation'] else reference_row(path.parent,s,m)
            r.update(split=s['study_split'],development_round=s.get('development_round'),pair_index=s['pair_index'],
                     layout_family=s['layout_family'],layout_label=s['layout_label'],swapped=s['swapped'])
            if s['policy_evaluation']:
                r.update(strict_pass=strict_success(r),failure_reasons=reasons(r))
                r['native_surface_flag_agrees']=r['strict_pass']==r['surface_goal_zero_spill_success']
                assert r['strict_pass']==(not r['failure_reasons'])
            attempts.append(r)
    cases=[]
    for s in p['test']:
        folder=folders.get(s['name']);r=next((x for x in attempts if x['name']==s['name']),None)
        if folder:
            actual=json.loads((folder/'parameters.json').read_text())
            assert all(actual[k]==v for k,v in s.items()),s['name']
        if r:assert r['checkpoint_sha256']==sha
        cases.append(r or dict(name=s['name'],status='incomplete' if folder else 'pending',expected_stop=s['expected_stop']))
    done=[r for r in cases if r['status']=='complete']
    counts=dict(planned=30,completed=len(done),pending=sum(r['status']=='pending' for r in cases),
        incomplete=sum(r['status']=='incomplete' for r in cases),strict_passes=sum(r['strict_pass'] for r in done),
        verified_grasps=sum(r['grasp_verified'] for r in done),visual_grasps=sum(r['visual_grasp_confirmed'] for r in done),
        actual_transfers=sum(r['delivered_to_cup_ml']>=1 for r in done),zero_measured_loss=sum(r['zero_spill'] for r in done),
        completed_return_sequences=sum(r['completed_return_release_sequence'] for r in done))
    result=dict(status='complete' if len(done)==30 else 'incomplete',counts=counts,cases=cases,all_attempts=attempts,incomplete_attempts=incomplete,
        demonstrations=[r for r in attempts if r['split'] in ('train','validation')],
        development=[r for r in attempts if r['split']=='development' and r['checkpoint_sha256']==sha],
        checkpoint_sha256=sha,scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        primary_definition='Verified physical and visual grasp, sustained bilateral contact >=80%, >=1 nominal mL transfer, correct full/empty stop, return/release/retreat, no gates, >=90% rim height or source<=max(.25mL,1% initial), zero final outside and ever-near-worktop particles.',
        caveat='20Hz simulator measurements, not calibrated physical volume or continuously splash-free performance. No-pour attempts cannot pass.')
    if (ROOT/'final-selection.json').exists():
        selection=json.loads((ROOT/'final-selection.json').read_text())
        assert selection['selected_checkpoint_sha256']==sha and selection['scene_plan_sha256']==result['scene_plan_sha256']
        result['selection']=selection
    if require_thirty:assert len(done)==30,'Keep every incomplete/pending case in the denominator'
    (ROOT/'wide-results.json').write_text(json.dumps(result,indent=2)+'\n')
    if len(done)==30:
        with (ROOT/'final-benchmark.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(done[0]),lineterminator='\n');writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in done)
    print(json.dumps(dict(counts=counts,accepted_references=sum(r['reference_accepted'] for r in result['demonstrations']),
                         development=[{k:r[k] for k in ('name','strict_pass','failure_reasons')} for r in result['development']]),indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-thirty',action='store_true');report(p.parse_args().require_thirty)
