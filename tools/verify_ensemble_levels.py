"""Verify actual frozen models on all development frames; no optimization or holdout use."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from fit_dynamic_pair import scores,passes
from repair_level_model_v2 import episode_mae
from ensemble_level_model import predict_ensemble_level


@torch.inference_mode()
def infer(data,artifact,device,batch=128):
    values=[];visibility=[]
    for i in range(0,len(data['x']),batch):
        p,v=predict_ensemble_level(data['x'][i:i+batch].to(device),data['xy'][i:i+batch].to(device),data['geometry'][i:i+batch],artifact)
        values.append(p.to('cuda'));visibility.append(v.to('cuda'))
    return torch.cat(values),torch.cat(visibility)


def main():
    started=time.monotonic();prefix='pair-r17'
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/(prefix+'-protocol.json')).exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    protected_unchanged();plan=read(ROOT/'ensemble-r17-plan.json');screen=read(ROOT/'ensemble-r17-screen.json')
    assert screen['development_accuracy_screen_passed'] and screen['plan_sha256']==sha(ROOT/'ensemble-r17-plan.json')
    assert all(sha(ROOT/n)==h for n,h in plan['frozen_files'].items())
    assert plan['weights']==[.5,.5] and plan['rgb_members']==['pair-r12-rgb.pt','pair-r14-rgb.pt'] and plan['color']=='pair-r14-color.pt'
    previous=read(ROOT/'pair-r14-protocol.json');assert all(sha(ROOT/n)==h for n,h in previous['frozen_files'].items())
    result=subprocess.run([sys.executable,'-m','unittest','-v','test_ensemble_levels'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    checks=dict(passed=result.returncode==0,output=result.stdout+result.stderr,source_sha256=sha(ROOT/'ensemble_level_model.py'),test_sha256=sha(ROOT/'test_ensemble_levels.py'))
    with (ROOT/'ensemble-contracts.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')
    assert checks['passed'];assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    files=set(previous['frozen_files'])|set(plan['frozen_files'])|{'ensemble-r17-plan.json','ensemble-r17-screen.json','ensemble_level_model.py','test_ensemble_levels.py','ensemble-contracts.json','verify_ensemble_levels.py','pair-r14-protocol.json','pair-r14.json','scale_level_cnn.py','fit_scale_level.py'}
    cfg=dict(round='r17',protocol_sha256=sha(ROOT/'protocol.json'),counts=previous['counts'],locator='tracker-t3.pt',weights=plan['weights'],optimizer_steps=0,
             method='Fixed equal-weight mean of separately supervised R12 direct RGB regression and R14 RGB relative-landmark prediction; independent R14 color-landmark cross-check. Member CNNs trained on the 4070. This packaging/verification performs no new optimization. Full-precision convolutions/matmul at inference.',frozen_files={n:sha(ROOT/n) for n in sorted(files)})
    with (ROOT/(prefix+'-protocol.json')).open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    meta=read(ROOT/'level-cache-v6-meta.json');g=read(ROOT/'protocol.json')['gates']
    with np.load(ROOT/'level-cache-v6.npz') as archive:arrays=dict(archive)
    data={s:dict(m,x=torch.tensor(arrays[s+'_x'],device='cuda'),xy=torch.tensor(arrays[s+'_xy'],device='cuda'),y=torch.tensor(arrays[s+'_y'],device='cuda')) for s,m in meta.items()}
    def load(name):return torch.load(ROOT/name,map_location='cpu',weights_only=False)
    artifacts=dict(rgb=dict(kind='rgb_equal_ensemble_v1',branch='rgb',weights=[.5,.5],members=[load(n) for n in plan['rgb_members']]),
                   color=dict(kind='full_precision_color_landmarks_v1',branch='color',member=load(plan['color'])))
    primary={};summaries={}
    for branch,artifact in artifacts.items():
        branch_start=time.monotonic();artifact.update(accepted=False,scope='Known-size simulated perception; no autonomous pouring acceptance',module_sha256=sha(ROOT/'ensemble_level_model.py'))
        predictions={};validity=[]
        for s,a in data.items():
            predictions[s],v=infer(a,artifact,'cuda');validity.append(bool(v.all()))
        tr,va=data['train'],data['validation'];training=scores(predictions['train'],tr);validation=scores(predictions['validation'],va)
        episodes={s:episode_mae(predictions[s],a) for s,a in data.items()};maximum=max(v for part in episodes.values() for v in part.values())
        disagreement=None if branch=='rgb' else max(float((predictions[s]-primary[s]).abs().max()) for s in data)
        if branch=='rgb':primary=predictions
        ok=all(validity) and maximum<g['level_validation_mae'] and passes(training,g) and passes(validation,g) and (disagreement is None or disagreement<g['crosscheck_max_disagreement'])
        selected=dict(epoch=0,passed=ok,training=training,validation=validation,per_episode_mae=episodes,max_episode_mae=maximum,max_primary_disagreement=disagreement,score=maximum/g['level_validation_mae'])
        cpu,cv=infer(va,artifact,'cpu');single,sv=infer(va,artifact,'cuda',batch=1)
        cpu_single,csv=infer(va,artifact,'cpu',batch=1)
        parity=max(float((cpu-predictions['validation']).abs().max()),float((cpu_single-single).abs().max()))
        pathdiff=max(float((single-predictions['validation']).abs().max()),float((cpu_single-cpu).abs().max()))
        cpu_scores=scores(cpu,va);single_scores=scores(single,va);cpu_single_scores=scores(cpu_single,va);absent_ok=None
        if branch=='color':
            blank,v=predict_ensemble_level(torch.full_like(va['x'][:32],180),va['xy'][:32],va['geometry'][:32],artifact)
            absent_ok=bool(not v.any() and torch.isnan(blank).all())
        accepted=bool(ok and parity<g['cpu_gpu_level_difference'] and pathdiff<g['cpu_gpu_level_difference'] and bool(cv.all()) and bool(sv.all()) and bool(csv.all()) and passes(cpu_scores,g) and passes(single_scores,g) and passes(cpu_single_scores,g) and max(episode_mae(cpu,va).values())<g['level_validation_mae'] and max(episode_mae(single,va).values())<g['level_validation_mae'] and max(episode_mae(cpu_single,va).values())<g['level_validation_mae'] and absent_ok is not False)
        artifact['accepted']=accepted;file=ROOT/(prefix+'-'+branch+'.pt');torch.save(artifact,file)
        members=artifact.get('members',[artifact.get('member')]);parameters=sum(sum(t.numel() for t in m['state'].values()) for m in members)
        rows=[dict(m,split=s,target=y,prediction=v,error=v-y) for s,a in data.items() for m,y,v in zip(a['meta'],a['target'],predictions[s].cpu().tolist())]
        report=dict(passed=accepted,selected=selected,history=[selected],rows=rows,cpu_scores=cpu_scores,single_frame_scores=single_scores,cpu_single_frame_scores=cpu_single_scores,cpu_gpu_difference=parity,inference_path_difference=pathdiff,absent_color_rejected=absent_ok,parameters=parameters,optimizer_steps=0,fit_seconds=0,verification_seconds=time.monotonic()-branch_start,checkpoint_sha256=sha(file),helper_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'ensemble_level_model.py'))
        with (ROOT/(prefix+'-'+branch+'.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        summaries[branch]={k:report[k] for k in ('passed','parameters','optimizer_steps','fit_seconds','verification_seconds','cpu_gpu_difference','inference_path_difference','checkpoint_sha256')}
        print(json.dumps(dict(branch=branch,max_episode_pp=maximum*100,**summaries[branch])),flush=True)
    assert all(sha(ROOT/n)==h for n,h in cfg['frozen_files'].items());protected_unchanged()
    pair=dict(passed=all(v['passed'] for v in summaries.values()),models=summaries,counts=cfg['counts'],method=cfg['method'],protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),scene_protocol_sha256=sha(ROOT/'protocol.json'),total_seconds=time.monotonic()-started,autonomy_accepted=False)
    with (ROOT/(prefix+'.json')).open('x') as f:json.dump(pair,f,indent=2);f.write('\n')
    with (ROOT/'ensemble-development.json').open('x') as f:json.dump(dict(passed=pair['passed'],pair_report_sha256=sha(ROOT/(prefix+'.json')),reserved_started=False),f,indent=2);f.write('\n')
    print(json.dumps(pair),flush=True)

if __name__=='__main__':main()
