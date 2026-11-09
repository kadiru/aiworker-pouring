"""Supervised native crop refinement on existing train/validation recordings."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import project,plane_point
from pour_cup_locator import CupLocator,centers
from pour_temporal_localization import consensus
from scale_data_admission import admit,balanced_frame_weights,collection_paths
from scale_rim_refiner import RimRefiner,coordinates,crop_native,locate_static
from scale_rim_augmentation import affine
ROOT=Path('/workspace/learning/A-scale-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

@torch.inference_mode()
def collect(coarse,round_name):
    plan=read(ROOT/'plan.json');m=read(ROOT/'candidate.json');ap,rp=collection_paths(ROOT,m)
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert read(rp)['accepted'] and read(rp)['audit_sha256']==sha(ap)
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    data={s:dict(images=[],target=[],origin=[],coarse=[],score=[],absent=[],meta=[],parity=[]) for s in ('train','validation')}
    manifest=[];calibration=None;preview=[]
    for spec,r,split in admit(plan,m,read(ap)):
        assert sha(Path(r['path']))==r['sha256']
        with h5py.File(r['path']) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[1024,1024])
            assert calibration is None or calibration==cal;calibration=cal
            z=1.04+spec['declared_cup_geometry']['rim_z_m'];truth=project(np.r_[spec['cup_world'][:2],z],cal)
            frames=[('obs',i) for i in range(12)]+[('next_obs',-1)]
            rgb=[decode(d[g+'/cup_detail_image'][i],'jpeg95') for g,i in frames]
            assert all(im.shape==(1024,1024,3) for im in rgb)
            small=np.stack([np.asarray(Image.fromarray(im).resize((384,384),Image.Resampling.BILINEAR)).copy() for im in rgb])
            x=torch.tensor(small,device='cuda');uv,scores=centers(coarse(x))
            center=uv.cpu().numpy()*(1024/384)
            # Same predeclared synthetic absence diagnostic, at the coarse input scale.
            negative=x.clone();px,py=np.rint(truth*384/1024).astype(int)
            negative[:,max(0,py-24):min(384,py+48),max(0,px-50):min(384,px+50)]=180
            _,absent=centers(coarse(negative));a=data[split]
            for j,(g,i) in enumerate(frames):
                crop,origin=crop_native(rgb[j],center[j]);a['images'].append(crop)
                a['target'].append(truth-origin);a['origin'].append(origin);a['coarse'].append(center[j])
                a['score'].append(float(scores[j]));a['absent'].append(float(absent[j]))
                a['meta'].append(dict(name=spec['name'],group=g,index=i,xy=spec['cup_world'][:2],z=z,
                    cup_height_scale=spec['cup_height_scale'],cup_diameter_scale=spec['cup_diameter_scale'],
                    level=r['label']['target'],truth_uv=truth.tolist()))
            if split=='validation':
                a['parity'].append((len(a['meta'])-1,rgb[-1]))
                error=float(np.linalg.norm(plane_point(center[0],z,cal)[:2]-spec['cup_world'][:2])*1000)
                preview.append((error,spec['name'],rgb[0],truth,center[0]))
        manifest.append(dict(name=spec['name'],split=split,path=r['path'],sha256=r['sha256']))
    diagnostic=[dict(name=name,error_mm=err,truth_uv=truth.tolist(),coarse_uv=center.tolist())
                for err,name,im,truth,center in sorted(preview,key=lambda x:-x[0])[:4]]
    # Preserve aspect ratio in diagnostic panels (256 native pixels centered in a 512-wide label cell).
    canvas=Image.new('RGB',(1024,568),'white');draw=ImageDraw.Draw(canvas)
    for j,(err,name,im,truth,coarse_uv) in enumerate(sorted(preview,key=lambda x:-x[0])[:4]):
        crop,origin=crop_native(im,coarse_uv,256);panel=Image.fromarray(crop);pd=ImageDraw.Draw(panel)
        for uv,col in ((truth,'lime'),(coarse_uv,'magenta')):
            x,y=uv-origin;pd.line((x-6,y,x+6,y),fill=col,width=2);pd.line((x,y-6,x,y+6),fill=col,width=2)
        canvas.paste(panel,((j%2)*512+128,(j//2)*284+28))
        draw.text(((j%2)*512+4,(j//2)*284+4),f'{name}: {err:.3f} mm',fill='black')
    canvas.save(ROOT/f'rim-{round_name}-diagnostic.png');write(f'rim-{round_name}-diagnostic.json',dict(rows=diagnostic,legend='green projected truth, magenta frozen coarse detector; native RGB crops; validation diagnostics only'))
    for a in data.values():
        a['x']=torch.tensor(np.stack(a.pop('images')),device='cuda')
        a['u']=torch.tensor(np.asarray(a['target']),device='cuda',dtype=torch.float32)
        a['weights']=torch.tensor(balanced_frame_weights(a['meta']),device='cuda')
    write('cup-locator-data-manifest.json',manifest)
    return plan,data,calibration

@torch.inference_mode()
def evaluate(model,a,cal,max_correction):
    local=torch.cat([coordinates(model(x)) for x in a['x'].split(32)]).cpu().numpy()
    uv=local+np.asarray(a['origin']);rows=[]
    for m,u,center,score,absent in zip(a['meta'],uv,a['coarse'],a['score'],a['absent']):
        correction=float(np.linalg.norm(u-center));score=score if correction<=max_correction else 0.
        error=float(np.linalg.norm(plane_point(u,m['z'],cal)[:2]-m['xy'])*1000)
        rows.append(dict(name=m['name'],group=m['group'],index=m['index'],uv=u.tolist(),score=score,
            synthetic_absent_score=absent,error_mm=error,correction_px=correction))
    return dict(max_error_mm=max(r['error_mm'] for r in rows),mean_error_mm=float(np.mean([r['error_mm'] for r in rows])),
        minimum_positive_score=min(r['score'] for r in rows),maximum_synthetic_absent_score=max(a['absent']),rows=rows)

def windows(result,meta,cal,threshold,config):
    out=[]
    for name in dict.fromkeys(m['name'] for m in meta):
        history=[];accepted=None;m=next(m for m in meta if m['name']==name)
        for r in result['rows']:
            if r['name']!=name or r['group']!='obs':continue
            history.append((r['uv'],r['score']));accepted=consensus(history,threshold,config)
            if accepted is not None:break
        error=None if accepted is None else float(np.linalg.norm(plane_point(accepted['uv'],m['z'],cal)[:2]-m['xy'])*1000)
        out.append(dict(name=name,confirmed=accepted is not None,error_mm=error,frames=len(history)))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r4','r5'),default='r4');args=parser.parse_args()
    protocol_path=ROOT/f'locator-refit-{args.round}-plan.json'
    assert not (ROOT/'cup-neural-localization.json').exists() and not (ROOT/'component-selection.json').exists()
    protocol=read(protocol_path);m=read(ROOT/'candidate.json');ap,rp=collection_paths(ROOT,m)
    assert protocol['plan_sha256']==sha(ROOT/'plan.json') and protocol['audit_sha256']==sha(ap) and protocol['review_sha256']==sha(rp)
    source=ROOT/protocol['coarse_checkpoint'];assert sha(source)==protocol['coarse_checkpoint_sha256']
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);torch.manual_seed(protocol['seed']);start=time.perf_counter()
    coarse_state=torch.load(source,map_location='cpu',weights_only=False)['state']
    coarse=CupLocator().cuda().eval();coarse.load_state_dict(coarse_state)
    plan,data,cal=collect(coarse,args.round);tr=data['train'];va=data['validation'];gate=plan['component_gates']
    model=RimRefiner().cuda()
    if protocol.get('fine_initialization'):
        init=ROOT/protocol['fine_initialization'];assert sha(init)==protocol['fine_initialization_sha256']
        model.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['bank']['state'])
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=1e-4)
    y,x=torch.meshgrid(torch.arange(64,device='cuda')*4+2,torch.arange(64,device='cuda')*4+2,indexing='ij')
    fitstart=time.perf_counter();best=None;attempt=None;history=[];steps=0
    for epoch in range(1,protocol['epochs']+1):
        model.train()
        if epoch in (81,121):
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.multinomial(tr['weights'],len(tr['x']),replacement=True).split(16):
            image=tr['x'][ids].float();target=tr['u'][ids].clone();n=len(ids)
            rng=np.random.default_rng(protocol['seed']+steps)
            if protocol.get('augmentation',{}).get('kind')=='affine':
                cfg=protocol['augmentation'];angle=np.deg2rad(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees'],n))
                scale=rng.uniform(cfg['scale_min'],cfg['scale_max'],n);aspect=np.exp(rng.uniform(-cfg['aspect_log_range'],cfg['aspect_log_range'],n))
                matrix=np.stack((np.cos(angle)*scale*aspect,-np.sin(angle)*scale/aspect,
                    np.sin(angle)*scale*aspect,np.cos(angle)*scale/aspect),1).reshape(n,2,2)
                shift=rng.uniform(-cfg['translation_pixels'],cfg['translation_pixels'],(n,2))
                image,target=affine(image,target,torch.tensor(matrix,device='cuda',dtype=torch.float32),torch.tensor(shift,device='cuda',dtype=torch.float32))
            else:
                dx,dy=rng.integers(-24,25,2).tolist();padded=F.pad(image.permute(0,3,1,2),(24,24,24,24),value=180)
                image=padded[:,:,24-dy:280-dy,24-dx:280-dx].permute(0,2,3,1).contiguous();target+=target.new_tensor([dx,dy])
            image=(image*(.9+.2*torch.rand(n,1,1,1,device='cuda'))).clamp(0,255)
            logits=model(image);g=torch.exp(-((x-target[:,0,None,None])**2+(y-target[:,1,None,None])**2)/32.)
            g=g/g.sum((1,2),keepdim=True)
            heat=-(g.flatten(1)*logits.flatten(1).log_softmax(1)).sum(1).mean()
            loss=heat+.02*F.smooth_l1_loss(coordinates(logits),target)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        if epoch%protocol['validation_interval']:continue
        model.eval();vr=evaluate(model,va,cal,protocol['max_correction_pixels']);tt=evaluate(model,tr,cal,protocol['max_correction_pixels'])
        minimum=min(vr['minimum_positive_score'],tt['minimum_positive_score']);maximum=max(vr['maximum_synthetic_absent_score'],tt['maximum_synthetic_absent_score']);threshold=(minimum+maximum)/2
        config=dict(window=20,votes=8,radius_px=4.,fraction=.6,score_floor=max(.01,threshold*.6))
        win=windows(vr,va['meta'],cal,threshold,config)+windows(tt,tr['meta'],cal,threshold,config)
        eligible=max(vr['max_error_mm'],tt['max_error_mm'])<gate['localization_max_error_mm'] and minimum-maximum>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in win)
        row=dict(epoch=epoch,validation_max_mm=vr['max_error_mm'],validation_mean_mm=vr['mean_error_mm'],training_max_mm=tt['max_error_mm'],absence_margin=minimum-maximum,eligible=eligible)
        history.append(row);print(json.dumps(row),flush=True)
        candidate=dict(epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},validation=vr,training=tt,windows=win,threshold=threshold,consensus=config)
        if attempt is None or vr['max_error_mm']<attempt['validation']['max_error_mm']:attempt=candidate
        if eligible and (best is None or vr['max_error_mm']<best['validation']['max_error_mm']):best=candidate
    fit_seconds=time.perf_counter()-fitstart
    chosen=best or attempt;model.load_state_dict(chosen['state']);model.eval()
    identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(chosen['state'].items()))).hexdigest()
    bank=dict(kind='native_rim_refiner',identity=identity,coarse_state=coarse_state,state=chosen['state'],crop_size=256,max_correction_pixels=protocol['max_correction_pixels'])
    parity=[];end_to_end=[]
    for index,image in va['parity']:
        cpu,cs=locate_static(image,bank,device='cpu');gpu,gs=locate_static(image,bank,device='cuda')
        parity.append(float(np.linalg.norm(cpu-gpu)));end_to_end.append(float(np.linalg.norm(gpu-chosen['validation']['rows'][index]['uv'])))
    passed=best is not None and max(parity)<.1 and max(end_to_end)<.01
    artifact=dict(bank=bank,calibration=cal,threshold=chosen['threshold'],consensus=chosen['consensus'],accepted=passed)
    torch.save(artifact,ROOT/'cup-neural-localization-attempt.pt')
    report=dict(passed=passed,scope='all_nine_declared_cup_sizes',epochs=protocol['epochs'],optimizer_steps=steps,
        learned_parameters=sum(p.numel() for p in model.parameters()),frozen_coarse_parameters=sum(p.numel() for p in coarse.parameters()),
        input_resolution=1024,native_crop_size=256,plan_sha256=sha(ROOT/'plan.json'),refit_protocol_sha256=sha(protocol_path),
        augmentation=protocol.get('augmentation',dict(kind='integer_translation')),augmentation_helper_sha256=sha(ROOT/'scale_rim_augmentation.py'),
        fine_initialization=protocol.get('fine_initialization'),fine_initialization_sha256=protocol.get('fine_initialization_sha256'),
        audit_sha256=sha(ap),manual_review_sha256=sha(rp),source_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'scale_rim_refiner.py'),
        coarse_checkpoint_sha256=sha(source),counts={s:dict(images=len(a['meta']),episodes=len({r['name'] for r in a['meta']})) for s,a in data.items()},
        history=history,best_attempt=dict(epoch=attempt['epoch'],**attempt['validation']),training_check=chosen['training'],
        selected=None if not passed else dict(epoch=chosen['epoch'],**chosen['validation']),temporal_windows=chosen['windows'],
        cpu_gpu_max_difference_pixels=max(parity),end_to_end_max_difference_pixels=max(end_to_end),
        attempt_checkpoint_sha256=sha(ROOT/'cup-neural-localization-attempt.pt'),fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,
        device=torch.cuda.get_device_name(),method='Supervised coarse-to-fine RGB localization. Frozen r3 coarse CNN selects native 256-pixel crops; 257-pixel receptive-field CNN predicts Gaussian rim-center heatmap with soft-argmax subpixel coordinates. Heatmap cross-entropy plus 0.02 smooth-L1 coordinate loss. Equal size/episode sampling, protocol-declared geometric and brightness augmentation. No ground-truth crops at inference; no RL or active model change.')
    if passed:
        torch.save(artifact,ROOT/'cup-neural-localization.pt');report['checkpoint_sha256']=sha(ROOT/'cup-neural-localization.pt')
        write('scope-selection.json',dict(passed=True,scope=report['scope'],epoch=chosen['epoch'],checkpoint_sha256=report['checkpoint_sha256'],
            max_validation_error_mm=chosen['validation']['max_error_mm'],max_training_error_mm=chosen['training']['max_error_mm'],threshold=chosen['threshold'],
            module_sha256=report['module_sha256'],refit_protocol_sha256=report['refit_protocol_sha256'],model_activated=False))
    write('cup-neural-localization.json',report)
    print(json.dumps(dict(passed=passed,epoch=chosen['epoch'],validation_max_mm=chosen['validation']['max_error_mm'],training_max_mm=chosen['training']['max_error_mm'],cpu_gpu_max_px=max(parity))),flush=True)
if __name__=='__main__':main()
