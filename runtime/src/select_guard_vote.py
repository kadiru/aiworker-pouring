"""CPU-only pre-test calibration; retain every learned tensor and motor path."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from pour_guard_vote_study import ROOT,plan
from pour_guard_actor import held_gripper,slow_empty_evidence

BASE=Path('/workspace/learning/A-guard-20261103-v1')
PARENT='f4ddb8b9f288ad2e5e8a9e6ac495f57a94c7413a0cdb4bbb040a89769bd36098'


def main():
    plan();assert not list(Path('/workspace/outputs/fountain').glob('*/*H_A_test*/parameters.json'))
    assert hashlib.sha256((BASE/'best.pt').read_bytes()).hexdigest()==PARENT
    b=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);cap=b['pour_primitive']['phase_cap_deg']
    gap=b['pour_primitive']['grasp_closure']['min_deficit_rad'];episodes=[];grasp_replays=[]
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*G_A_*/metrics.json')):
        m=json.loads(p.read_text())
        if m['policy']['checkpoint_sha256']!=PARENT:continue
        s=json.loads((p.parent/'parameters.json').read_text());d=json.loads((p.parent/'policy-decisions.json').read_text())
        rows=json.loads((p.parent/'samples.json').read_text());episodes.append((s,d,rows))
        history=[];physical=0;first=None
        for x in d:
            if x['mode']!='Visual grasp: verify':continue
            c=x['grasp_check'];hard=bool(c['finger_obstruction'] and c['arm_lift_m']>.035)
            physical=physical+1 if hard else 0
            history=(history+[bool(c['score']>.23 and c['moved'] and c['follows'])])[-12:] if hard else []
            if physical>=6 and sum(history)>=6:first=x['t'];break
        grasp_replays.append(dict(name=s['name'],window_confirmation_time=first,
            original_confirmed=any(x['visual_grasp_confirmed'] for x in d)))
    sweep=[]
    for probability in (.90,.92,.93,.94,.95,.97):
        thresholds=dict(slow_empty_probability=probability,slow_empty_max_flow_ml_s=1.,empty_cap_dwell_seconds=.25)
        calls=[];eligible_negative_frames=0
        for s,d,rows in episodes:
            started=None;votes=0;first_empty=next((x['t'] for x,r in zip(d,rows) if r['bottle_ml']<=max(.25,.01*s['initial_ml'])),None)
            for x,r in zip(d,rows):
                if not x['visual_grasp_confirmed'] or x['stop_latched'] is not None:continue
                if x['own_pour_phase_deg']>=cap-.25 and started is None:started=x['t']
                q=np.radians(r['joint_positions_deg'])
                kw=dict(threshold=thresholds,phase=x['own_pour_phase_deg'],cap=cap,tilt=x['estimated_tilt_from_joints'],
                    cap_age=None if started is None else x['t']-started,held=held_gripper(q,gap),
                    present=x['source_present_probability'],empty=x['empty_probability'],flow_ml_s=x['visual_flow_fraction_s']*72.8)
                empty=r['bottle_ml']<=max(.25,.01*s['initial_ml'])
                if slow_empty_evidence(**dict(kw,empty=1.)) and not empty:eligible_negative_frames+=1
                votes=votes+1 if slow_empty_evidence(**kw) else 0
                if votes>=24:
                    hold=[v for v in rows if 3.25<=v['t']<=x['t']]
                    fraction=sum(v['both_fingers'] for v in hold)/max(1,len(hold))
                    calls.append(dict(name=s['name'],t=x['t'],source_ml=r['bottle_ml'],correct_empty=empty,
                        latency_from_empty_s=None if first_empty is None else x['t']-first_empty,
                        bilateral_fraction_until_call=fraction));break
        target=next((x for x in calls if x['name']=='G_A_test_04'),None)
        sweep.append(dict(probability=probability,calls=calls,false_positive=sum(not x['correct_empty'] for x in calls),
            eligible_negative_frames=eligible_negative_frames,
            fixes_target=bool(target and target['correct_empty'] and target['latency_from_empty_s']<=2. and target['bilateral_fraction_until_call']>=.8)))
    chosen=next((x for x in sweep if x['false_positive']==0 and x['fixes_target']),None)
    assert chosen,'No calibrated safe/fast slow-empty point; inspect consumed traces'
    assert all(x['window_confirmation_time'] is not None for x in grasp_replays if x['original_confirmed'])
    assert next(x for x in grasp_replays if x['name']=='G_A_test_00')['window_confirmation_time'] is not None
    thresholds=dict(b['thresholds']);thresholds.update(grasp_visual_window=12,verify_seconds=1.25,
        slow_empty_probability=chosen['probability'],slow_empty_votes=24,empty_cap_dwell_seconds=.25,slow_empty_max_flow_ml_s=1.)
    note=dict(parent_checkpoint_sha256=PARENT,method='Controller-only temporal and confidence calibration on consumed G_A records',
        trained_tensors_and_motor_paths_unchanged=True,grasp_replays=grasp_replays,slow_empty_sweep=sweep,
        selected=chosen,thresholds=thresholds,
        selection='Lowest slow confidence with zero observed premature cap latches, correct target detection within2s, and >=80% bilateral contact until call;24votes plus physical/cap/flow gates. Support is limited; new native tests are required.',
        no_H_A_fit_or_test_data_used=True,cuda_initialized=torch.cuda.is_initialized())
    b['thresholds']=thresholds;b['model_version']='visual_guard_hierarchy_v11';b['controller_parent_checkpoint_sha256']=PARENT
    b['vote_calibration']=note;b['scene_plan_sha256']=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest()
    b['sources']={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in b['sources']}
    b['method']='Visual servo with continuous physical hold and windowed RGB confirmation; trained normalized cup model and contextual motion; bounded full feedback and calibrated slow empty confirmation'
    torch.save(b,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    parent=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);after=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(parent[k][n],after[k][n]) for k in ('cup_model','source_detector') for n in parent[k])
    assert parent['pour_primitive']==after['pour_primitive'] and parent['primitive']==after['primitive']
    for name in ('data-manifest.json','training-history.json','localization-validation.json'):
        shutil.copy2(BASE/name,ROOT/name)
    note['selected_checkpoint_sha256']=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest()
    (ROOT/'vote-calibration.json').write_text(json.dumps(note,indent=2)+'\n')
    print(json.dumps(dict(checkpoint_sha256=note['selected_checkpoint_sha256'],selected=chosen,cuda_initialized=torch.cuda.is_initialized()),indent=2))


if __name__=='__main__':main()
