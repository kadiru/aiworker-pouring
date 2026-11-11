"""Train independent relative-landmark level models; retain every original gate."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged
from fit_dynamic_pair import scores, passes
from scale_data_admission import balanced_frame_weights
from repair_level_model import color_outline
from repair_level_model_v2 import episode_mae
from landmark_level_model import LandmarkLevelNet, points_from_logits, level_from_points, predict_landmark_level


@torch.inference_mode()
def infer(model, a, calibration):
    result=[]
    for i in range(0, len(a['input']), 128):
        points=points_from_logits(model(a['input'][i:i+128]))
        result.append(level_from_points(points,a['xy'][i:i+128],a['geometry'][i:i+128],calibration).clamp(0,1))
    return torch.cat(result)


def main():
    prefix='pair-r14';cfg=read(ROOT/(prefix+'-protocol.json'));started=time.monotonic()
    assert not (ROOT/(prefix+'.json')).exists() and not (ROOT/'component-selection.json').exists()
    assert all(sha(ROOT/n)==h for n,h in cfg['frozen_files'].items())
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);protected_unchanged()
    g=read(ROOT/'protocol.json')['gates'];meta=read(ROOT/'level-cache-v6-meta.json')
    with np.load(ROOT/'level-cache-v6.npz') as archive:arrays=dict(archive)
    with np.load(ROOT/'landmark-level-labels.npz') as archive:labels=dict(archive)
    with np.load(ROOT/'native-shift-crops.npz') as archive:native=torch.tensor(archive['images'],device='cuda')
    data={}
    for split,m in meta.items():
        a=dict(m)
        for k in ('x','xy','y'):a[k]=torch.tensor(arrays[split+'_'+k],device='cuda')
        a['points']=torch.tensor(labels[split+'_points'],device='cuda')
        a['xy_choices']=torch.tensor(labels[split+'_xy_choices'],device='cuda')
        data[split]=a
    tr,va=data['train'],data['validation'];weights=np.zeros(len(tr['x']))
    for moving in (False,True):
        ids=[i for i,m in enumerate(tr['meta']) if bool(m.get('moving'))==moving]
        weights[ids]=np.asarray(balanced_frame_weights([tr['meta'][i] for i in ids]))*.5
    weights=torch.tensor(weights,device='cuda')
    yy,xx=torch.meshgrid(torch.arange(24,device='cuda')*4+2,torch.arange(24,device='cuda')*4+2,indexing='ij')
    summaries={};primary={}
    for kind in ('rgb','color'):
        torch.manual_seed(cfg['seed']+(kind=='color'));model=LandmarkLevelNet().cuda()
        for a in data.values():
            a['input']=a['x']
            if kind=='color':a['input'],valid=color_outline(a['x']);assert valid.all()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        best=attempt=None;history=[];steps=0;fit_started=time.monotonic()
        for epoch in range(1,cfg['epochs']+1):
            assert time.monotonic()-started<cfg['wall_seconds']
            model.train()
            if epoch in cfg['lr_steps']:
                for group in optimizer.param_groups:group['lr']/=3
            for ids in torch.multinomial(weights,len(tr['x']),replacement=True).split(cfg['batch_size']):
                n=len(ids);choice=torch.randint(0,3,(n,),device='cuda');alternative=choice>0
                image=tr['x'][ids].clone();image[alternative]=native[ids[alternative],choice[alternative]-1]
                if kind=='color':image,valid=color_outline(image);assert valid.all()
                target_points=tr['points'][ids,choice];xy=tr['xy_choices'][ids,choice]
                geometry=[tr['geometry'][i] for i in ids.cpu().tolist()]
                logits=model(image);points=points_from_logits(logits)
                gaussian=(-((xx-target_points[:,:,0,None,None])**2+(yy-target_points[:,:,1,None,None])**2)/32).flatten(2).softmax(2)
                heatmap_loss=-(gaussian*logits.flatten(2).log_softmax(2)).sum(2).mean()
                level=level_from_points(points,xy,geometry,cfg['calibration'])
                squared=(level-tr['y'][ids]).square()
                loss=heatmap_loss+cfg['point_weight']*F.smooth_l1_loss(points,target_points)+cfg['level_weight']*(squared.mean()+cfg['top_quartile_weight']*squared.topk(max(1,n//4)).values.mean())
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();steps+=1
            if epoch%10:continue
            model.eval();tp=infer(model,tr,cfg['calibration']);vp=infer(model,va,cfg['calibration'])
            training=scores(tp,tr);validation=scores(vp,va)
            disagreement=None if kind=='rgb' else max(float((tp-primary['train']).abs().max()),float((vp-primary['validation']).abs().max()))
            episodes=dict(train=episode_mae(tp,tr),validation=episode_mae(vp,va));maximum=max(v for part in episodes.values() for v in part.values())
            ok=maximum<g['level_validation_mae'] and passes(training,g) and passes(validation,g) and (disagreement is None or disagreement<g['crosscheck_max_disagreement'])
            ratios=[value/limit for group in (training,validation) for m in group.values() for s in m['sizes'].values() for value,limit in ((s['mae'],g['level_validation_mae']),(s['near_max'],g['level_validation_near_max_error']),(s['near_overestimate'],g['level_validation_near_overestimate']))]
            ratios.append(maximum/g['level_validation_mae'])
            if disagreement is not None:ratios.append(disagreement/g['crosscheck_max_disagreement'])
            score=max(ratios)+.1*validation['all']['mae']/g['level_validation_mae']
            row=dict(epoch=epoch,passed=ok,score=score,training=training,validation=validation,per_episode_mae=episodes,max_episode_mae=maximum,max_primary_disagreement=disagreement);history.append(row)
            chosen=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},row=row)
            if attempt is None or score<attempt['row']['score']:attempt=chosen
            if ok and (best is None or score<best['row']['score']):best=chosen
            if epoch%100==0:print(json.dumps(dict(branch=kind,epoch=epoch,passed=ok,score=score,max_episode_pp=maximum*100,disagreement=disagreement)),flush=True)
            torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),best=best,attempt=attempt,protocol_sha256=sha(ROOT/(prefix+'-protocol.json'))),ROOT/(prefix+'-'+kind+'-progress.pt'))
        chosen=best or attempt;model.load_state_dict(chosen['state']);model.eval()
        identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(chosen['state'].items()))).hexdigest()
        artifact=dict(kind='geometry_landmarks_v1',branch=kind,state=chosen['state'],identity=identity,accepted=False,calibration=cfg['calibration'],module_sha256=sha(ROOT/'landmark_level_model.py'),scope='Known-size simulated perception; no autonomous pouring acceptance')
        predictions={s:infer(model,a,cfg['calibration']) for s,a in data.items()}
        if kind=='rgb':primary=predictions
        gpu=[];cpu=[];visible=[];cpu_visible=[]
        for i in range(0,len(va['x']),128):
            q,v=predict_landmark_level(va['x'][i:i+128],va['xy'][i:i+128],va['geometry'][i:i+128],artifact);gpu.append(q);visible.append(v)
            q,v=predict_landmark_level(va['x'][i:i+128].cpu(),va['xy'][i:i+128].cpu(),va['geometry'][i:i+128],artifact);cpu.append(q);cpu_visible.append(v)
        gpu=torch.cat(gpu);cpu=torch.cat(cpu);parity=float((cpu-gpu.cpu()).abs().max());pathdiff=float((gpu-predictions['validation']).abs().max())
        cpu_scores=scores(cpu.cuda(),va);absent_ok=None
        if kind=='color':
            blank,visible_blank=predict_landmark_level(torch.full_like(va['x'][:32],180),va['xy'][:32],va['geometry'][:32],artifact)
            absent_ok=bool(not visible_blank.any() and torch.isnan(blank).all())
        accepted=best is not None and parity<g['cpu_gpu_level_difference'] and pathdiff<1e-4 and passes(cpu_scores,g) and max(episode_mae(cpu.cuda(),va).values())<g['level_validation_mae'] and absent_ok is not False and all(bool(v.all()) for v in visible+cpu_visible)
        artifact['accepted']=accepted;file=ROOT/(prefix+'-'+kind+'.pt');torch.save(artifact,file)
        rows=[]
        for split,a in data.items():
            rows += [dict(m,split=split,target=y,prediction=v,error=v-y) for m,y,v in zip(a['meta'],a['target'],predictions[split].cpu().tolist())]
        report=dict(passed=accepted,selected=chosen['row'],history=history,rows=rows,cpu_scores=cpu_scores,cpu_gpu_difference=parity,inference_path_difference=pathdiff,absent_color_rejected=absent_ok,parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,fit_seconds=time.monotonic()-fit_started,checkpoint_sha256=sha(file),helper_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'landmark_level_model.py'))
        (ROOT/(prefix+'-'+kind+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        summaries[kind]={k:report[k] for k in ('passed','parameters','optimizer_steps','fit_seconds','cpu_gpu_difference','inference_path_difference','checkpoint_sha256')}
        print(json.dumps(dict(branch=kind,selected_epoch=chosen['row']['epoch'],**summaries[kind])),flush=True)
    assert all(sha(ROOT/n)==h for n,h in cfg['frozen_files'].items());protected_unchanged()
    report=dict(passed=all(v['passed'] for v in summaries.values()),models=summaries,counts=cfg['counts'],protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),scene_protocol_sha256=sha(ROOT/'protocol.json'),method=cfg['method'],total_seconds=time.monotonic()-started,autonomy_accepted=False)
    with (ROOT/(prefix+'.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
