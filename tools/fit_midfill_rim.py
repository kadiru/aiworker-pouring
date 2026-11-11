"""Expanded initial-frame rim supervision. Temporal tracking is evaluated separately.

All 12 initial frames are supervised for all 279 episodes. No moving-frame
tracking gate is removed: tracker-t3 replay covers every development sequence.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import project
from pour_cup_locator import CupLocator,centers
from scale_data_admission import admit,balanced_frame_weights,collection_paths
from scale_rim_refiner import RimRefiner,coordinates,crop_native,locate_static
from scale_rim_augmentation import affine
from fit_scale_rim_refiner import evaluate,windows
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protected_unchanged():
    m=read(ROOT/'candidate.json')
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())

@torch.inference_mode()
def collect(coarse,coarse_only=False,manifest_name='rim-d0-data.json'):
    from perception_midfill_data import entries as reviewed_entries
    entries=reviewed_entries()
    data={s:dict(images=[],target=[],origin=[],coarse=[],score=[],absent=[],meta=[],parity=[]) for s in ('train','validation')}
    calibration=None;manifest=[]
    for spec,r,split in entries:
        path=Path(r['path']);assert sha(path)==r['sha256']
        actual=read(path.parent/'parameters.json');assert all(actual[k]==v for k,v in spec.items())
        with h5py.File(path) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[1024,1024])
            assert calibration is None or calibration==cal;calibration=cal
            z=1.04+spec['declared_cup_geometry']['rim_z_m'];truth=project(np.r_[spec['cup_world'][:2],z],cal)
            frames=[('obs',i) for i in range(12)]
            rgb=[decode(d[g+'/cup_detail_image'][i],'jpeg95') for g,i in frames]
            small=np.stack([np.asarray(Image.fromarray(im).resize((384,384),Image.Resampling.BILINEAR)).copy() for im in rgb])
            x=torch.tensor(small,device='cuda');uv,scores=centers(coarse(x));center=uv.cpu().numpy()*(1024/384)
            negative=x.clone();px,py=np.rint(truth*384/1024).astype(int)
            negative[:,max(0,py-24):min(384,py+48),max(0,px-50):min(384,px+50)]=180
            _,absent=centers(coarse(negative));a=data[split]
            for j,(g,i) in enumerate(frames):
                crop,origin=crop_native(rgb[j],center[j]);a['images'].append(small[j] if coarse_only else crop);a['target'].append(truth*(384/1024) if coarse_only else truth-origin);a['origin'].append(origin);a['coarse'].append(center[j])
                a['score'].append(float(scores[j]));a['absent'].append(float(absent[j]))
                a['meta'].append(dict(name=spec['name'],group=g,index=i,xy=spec['cup_world'][:2],z=z,cup_height_scale=spec['cup_height_scale'],cup_diameter_scale=spec['cup_diameter_scale'],moving=bool(spec.get('calibration_motion')),truth_uv=truth.tolist()))
            if split=='validation':a['parity'].append((len(a['meta'])-1,rgb[-1]))
        manifest.append(dict(name=spec['name'],split=split,sha256=r['sha256'],path=r['path'],frames=len(frames)))
    for a in data.values():
        a['x']=torch.tensor(np.stack(a.pop('images')),device='cuda');a['u']=torch.tensor(np.asarray(a['target']),device='cuda',dtype=torch.float32)
        a['weights']=torch.tensor(balanced_frame_weights(a['meta']),device='cuda')
    (ROOT/manifest_name).write_text(json.dumps(manifest,indent=2)+'\n')
    return data,calibration

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('d4',),default='d4');args=parser.parse_args();prefix='rim-'+args.round
    cfg=read(ROOT/(prefix+'-protocol.json'));assert not (ROOT/(prefix+'.json')).exists() and not (ROOT/'component-selection.json').exists()
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    protected_unchanged()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);torch.manual_seed(cfg['seed']);started=time.perf_counter()
    parent=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False);bank=parent['bank']
    coarse_state=bank['coarse_state']
    if cfg.get('coarse_checkpoint'):coarse_state=torch.load(ROOT/cfg['coarse_checkpoint'],map_location='cpu',weights_only=False)['state']
    coarse=CupLocator().cuda().eval();coarse.load_state_dict(coarse_state);data,cal=collect(coarse,manifest_name=prefix+'-data.json')
    assert all(bool(((a['u']>16)&(a['u']<240)).all()) for a in data.values()),'Refinement targets outside usable crop; no frames may be dropped'
    tr,va=data['train'],data['validation'];model=RimRefiner().cuda()
    initial=bank['state'] if not cfg.get('fine_initialization') else torch.load(ROOT/cfg['fine_initialization'],map_location='cpu',weights_only=False)['bank']['state']
    model.load_state_dict(initial)
    if cfg.get('balance_motion_groups'):
        weights=np.zeros(len(tr['meta']))
        for moving in (False,True):
            ids=[i for i,m in enumerate(tr['meta']) if bool(m['moving'])==moving]
            weights[ids]=np.asarray(balanced_frame_weights([tr['meta'][i] for i in ids]))*.5
        assert abs(weights.sum()-1)<1e-6;tr['weights']=torch.tensor(weights,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    y,x=torch.meshgrid(torch.arange(64,device='cuda')*4+2,torch.arange(64,device='cuda')*4+2,indexing='ij')
    best=attempt=None;history=[];steps=0;gate=read(ROOT/'protocol.json')['gates'];fit_started=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        assert time.perf_counter()-started<cfg['wall_seconds'], 'Frozen fit deadline reached'
        model.train()
        if epoch in cfg['lr_steps']:
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.multinomial(tr['weights'],len(tr['x']),replacement=True).split(16):
            image=tr['x'][ids].float();target=tr['u'][ids].clone();n=len(ids)
            angle=(torch.rand(n,device='cuda')*2-1)*np.deg2rad(3);scale=.98+.04*torch.rand(n,device='cuda')
            matrix=torch.stack((angle.cos()*scale,-angle.sin()*scale,angle.sin()*scale,angle.cos()*scale),1).reshape(n,2,2)
            image,target=affine(image,target,matrix,(torch.rand(n,2,device='cuda')*2-1)*8)
            image=(image*(.95+.1*torch.rand(n,1,1,1,device='cuda'))).clamp(0,255)
            logits=model(image);g=(-((x-target[:,0,None,None])**2+(y-target[:,1,None,None])**2)/32).flatten(1).softmax(1).reshape(-1,64,64)
            loss=-(g.flatten(1)*logits.flatten(1).log_softmax(1)).sum(1).mean()+.02*F.smooth_l1_loss(coordinates(logits),target)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        if epoch%5:continue
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),protocol_sha256=sha(ROOT/(prefix+'-protocol.json'))),ROOT/(prefix+'-progress.pt'))
        model.eval();tt=evaluate(model,tr,cal,bank['max_correction_pixels']);vr=evaluate(model,va,cal,bank['max_correction_pixels'])
        minimum=min(tt['minimum_positive_score'],vr['minimum_positive_score']);maximum=max(tt['maximum_synthetic_absent_score'],vr['maximum_synthetic_absent_score']);threshold=(minimum+maximum)/2
        config=dict(parent['consensus'],score_floor=max(.01,threshold*.6))
        win=windows(tt,tr['meta'],cal,threshold,config)+windows(vr,va['meta'],cal,threshold,config)
        ok=max(tt['max_error_mm'],vr['max_error_mm'])<gate['localization_max_error_mm'] and minimum-maximum>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in win)
        row=dict(epoch=epoch,passed=ok,train_max_mm=tt['max_error_mm'],validation_max_mm=vr['max_error_mm'],absence_margin=minimum-maximum,confirmed=sum(w['confirmed'] for w in win),windows=len(win));history.append(row);print(json.dumps(row),flush=True)
        result=dict(row=row,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},training=tt,validation=vr,windows=win,threshold=threshold,consensus=config)
        if attempt is None or row['validation_max_mm']<attempt['row']['validation_max_mm']:attempt=result
        if ok and (best is None or row['validation_max_mm']<best['row']['validation_max_mm']):best=result
    chosen=best or attempt;identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(chosen['state'].items()))).hexdigest()
    identity=hashlib.sha256((identity+sha(ROOT/cfg['coarse_checkpoint'])).encode()).hexdigest() if cfg.get('coarse_checkpoint') else identity
    newbank=dict(kind='native_rim_refiner',identity=identity,coarse_state=coarse_state,state=chosen['state'],crop_size=bank['crop_size'],max_correction_pixels=bank['max_correction_pixels'])
    parity=[];shared=[]
    for i,image in va['parity']:
        cpu,_=locate_static(image,newbank,device='cpu');gpu,_=locate_static(image,newbank,device='cuda');parity.append(float(np.linalg.norm(cpu-gpu)));shared.append(float(np.linalg.norm(gpu-chosen['validation']['rows'][i]['uv'])))
    passed=best is not None and max(parity)<.1 and max(shared)<.01
    artifact=dict(bank=newbank,calibration=cal,threshold=chosen['threshold'],consensus=chosen['consensus'],accepted=passed)
    torch.save(artifact,ROOT/(prefix+'.pt'));chosen.pop('state')
    report=dict(passed=passed,selected=chosen,history=history,counts={s:dict(episodes=len({m['name'] for m in a['meta']}),frames=len(a['meta'])) for s,a in data.items()},parameters=sum(p.numel() for p in model.parameters()),steps=steps,cpu_gpu_max_pixels=max(parity),shared_path_max_pixels=max(shared),checkpoint_sha256=sha(ROOT/(prefix+'.pt')),protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),fit_seconds=time.perf_counter()-fit_started,total_seconds=time.perf_counter()-started,autonomy_accepted=False)
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    protected_unchanged()
    (ROOT/(prefix+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=passed,selected=chosen['row'])),flush=True)
if __name__=='__main__':main()
