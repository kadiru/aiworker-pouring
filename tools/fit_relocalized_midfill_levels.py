"""Finite 4070-only fit of independent crop-tolerant level models."""
import argparse, hashlib, json, time
import numpy as np
import torch
from dynamic_level_data import ROOT, sha, read
from fit_dynamic_pair import scores, passes, infer
from scale_data_admission import balanced_frame_weights
from scale_level_cnn import level_context
from scale_rim_augmentation import affine
from repair_level_model import OutlineLevelNet, color_outline
from repair_level_model_v2 import predict_repair, repair_context
from repair_level_model_v2 import episode_mae


def main():
    p=argparse.ArgumentParser();p.add_argument('--round',choices=['r12'],default='r12');a=p.parse_args()
    prefix='pair-'+a.round;cfg=read(ROOT/(prefix+'-protocol.json'));started=time.monotonic()
    assert not (ROOT/(prefix+'.json')).exists() and not (ROOT/'component-selection.json').exists()
    assert cfg['protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    g=read(ROOT/'protocol.json')['gates'];meta=read(ROOT/'level-cache-v6-meta.json');arrays=np.load(ROOT/'level-cache-v6.npz');data={}
    for split,m in meta.items():
        b=dict(m)
        for name in ('x','xy','y'):b[name]=torch.tensor(arrays[split+'_'+name],device='cuda')
        data[split]=b
    tr,va=data['train'],data['validation'];weights=np.zeros(len(tr['meta']))
    for moving in (False,True):
        ids=[i for i,m in enumerate(tr['meta']) if bool(m.get('moving'))==moving]
        weights[ids]=np.asarray(balanced_frame_weights([tr['meta'][i] for i in ids]))*.5
    weights=torch.tensor(weights,device='cuda');summaries={};primary={}
    for kind in ('rgb','color'):
        torch.manual_seed(cfg['seed']+(kind=='color'));model=OutlineLevelNet().cuda()
        model.load_state_dict(torch.load(ROOT/cfg['initialization'][kind],map_location='cpu',weights_only=False)['state'])
        for b in data.values():
            b['input']=b['x']
            if kind=='color':
                b['input'],valid=color_outline(b['x']);assert valid.all()
            b['context']=repair_context(b['input'],b['geometry'],b['xy'])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        best=attempt=None;history=[];steps=0;fit_start=time.monotonic()
        for epoch in range(1,cfg['epochs']+1):
            assert time.monotonic()-started<cfg['wall_seconds'],'Fit exceeded its frozen wall-clock budget'
            model.train()
            if epoch in cfg['lr_steps']:
                for group in opt.param_groups:group['lr']/=3
            for ids in torch.multinomial(weights,len(tr['x']),replacement=True).split(cfg['batch_size']):
                x=tr['input'][ids].float();n=len(ids)
                shift=(torch.rand(n,2,device='cuda')*2-1)*cfg['jitter_pixels']
                x,_=affine(x,x.new_zeros(n,2),torch.eye(2,device='cuda').repeat(n,1,1),shift)
                error=model(x,tr['context'][ids])-tr['y'][ids];sq=error.square()
                loss=sq.mean()+cfg['top_quartile_weight']*sq.topk(max(1,n//4)).values.mean()
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
            if epoch%10:continue
            model.eval()
            with torch.inference_mode():
                tp=infer(model,tr);vp=infer(model,va);training=scores(tp,tr);validation=scores(vp,va)
                diff=None if kind=='rgb' else max(float((tp-primary['train']).abs().max()),float((vp-primary['validation']).abs().max()))
                episodes=dict(train=episode_mae(tp,tr),validation=episode_mae(vp,va))
                episode_max=max(v for part in episodes.values() for v in part.values())
                ok=episode_max<g['level_validation_mae'] and passes(training,g) and passes(validation,g) and (diff is None or diff<g['crosscheck_max_disagreement'])
                ratios=[value/limit for group in (training,validation) for m in group.values() for s in m['sizes'].values() for value,limit in ((s['mae'],g['level_validation_mae']),(s['near_max'],g['level_validation_near_max_error']),(s['near_overestimate'],g['level_validation_near_overestimate']))]
                ratios.append(episode_max/g['level_validation_mae'])
                if diff is not None:ratios.append(diff/g['crosscheck_max_disagreement'])
                score=max(ratios)+.1*validation['all']['mae']/g['level_validation_mae']
                row=dict(epoch=epoch,per_episode_mae=episodes,max_episode_mae=episode_max,training=training,validation=validation,max_primary_disagreement=diff,passed=ok,score=score);history.append(row)
                choice=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},row=row)
                if attempt is None or score<attempt['row']['score']:attempt=choice
                if ok and (best is None or score<best['row']['score']):best=choice
            if epoch%100==0:print(json.dumps(dict(branch=kind,epoch=epoch,passed=ok,score=score,disagreement=diff)),flush=True)
            torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),best=best,attempt=attempt,protocol_sha256=sha(ROOT/(prefix+'-protocol.json'))),ROOT/(prefix+'-'+kind+'-progress.pt'))
        choice=best or attempt;model.load_state_dict(choice['state']);model.eval()
        identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(choice['state'].items()))).hexdigest()
        artifact=dict(kind='crop_tolerant_outline_v2',branch=kind,identity=identity,state=choice['state'],accepted=False,scope='development_only_known_cup_sizes_no_autonomy',module_sha256=sha(ROOT/'repair_level_model_v2.py'))
        with torch.inference_mode():
            gpu,gv=predict_repair(va['x'],va['xy'],va['geometry'],artifact);cpu,cv=predict_repair(va['x'].cpu(),va['xy'].cpu(),va['geometry'],artifact)
            parity=float((cpu-gpu.cpu()).abs().max());pathdiff=float((gpu-infer(model,va)).abs().max());cpu_scores=scores(cpu.cuda(),va)
            absent_ok=None
            if kind=='color':
                absent,visible=predict_repair(torch.full_like(va['x'],180),va['xy'],va['geometry'],artifact)
                absent_ok=bool(gv.all() and cv.all() and not visible.any() and torch.isnan(absent).all())
            if kind=='rgb':primary={s:infer(model,b) for s,b in data.items()}
            rows=[]
            for split,b in data.items():
                pred=infer(model,b).cpu().tolist()
                rows += [dict(m,split=split,target=y,prediction=v,error=v-y) for m,y,v in zip(b['meta'],b['target'],pred)]
        accepted=best is not None and parity<g['cpu_gpu_level_difference'] and pathdiff<1e-4 and passes(cpu_scores,g) and max(episode_mae(cpu.cuda(),va).values())<g['level_validation_mae'] and absent_ok is not False
        artifact['accepted']=accepted;file=ROOT/(prefix+'-'+kind+'.pt');torch.save(artifact,file)
        report=dict(passed=accepted,selected=choice['row'],history=history,rows=rows,cpu_scores=cpu_scores,cpu_gpu_difference=parity,inference_path_difference=pathdiff,absent_color_rejected=absent_ok,parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,fit_seconds=time.monotonic()-fit_start,checkpoint_sha256=sha(file),helper_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'repair_level_model_v2.py'))
        (ROOT/(prefix+'-'+kind+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        summaries[kind]={k:report[k] for k in ('passed','parameters','optimizer_steps','fit_seconds','cpu_gpu_difference','inference_path_difference','checkpoint_sha256')}
        print(json.dumps(dict(branch=kind,selected_epoch=choice['row']['epoch'],**summaries[kind])),flush=True)
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    m=read(ROOT/'candidate.json');assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    result=dict(passed=all(v['passed'] for v in summaries.values()),models=summaries,counts=read(ROOT/'level-cache-v6.json')['counts'],protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),scene_protocol_sha256=sha(ROOT/'protocol.json'),total_seconds=time.monotonic()-started,method=cfg['method'],autonomy_accepted=False)
    (ROOT/(prefix+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

from pathlib import Path
if __name__=='__main__':main()
