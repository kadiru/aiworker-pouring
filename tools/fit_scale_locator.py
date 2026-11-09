"""Separate nine-size locator refit; balanced episodes, all-size selection, no activation."""
import argparse,copy,hashlib,json,sys,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import project,plane_point
from pour_cup_locator import CupLocator,centers
from scale_data_admission import admit,balanced_frame_weights
ROOT=Path('/workspace/learning/A-scale-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def collect(dense=False):
    plan=read(ROOT/'plan.json');candidate=read(ROOT/'candidate.json');audit=read(ROOT/'collect-r0-audit.json')
    review=read(ROOT/'collect-r0-manual-review.json')
    assert review['accepted'] and review['audit_sha256']==sha(ROOT/'collect-r0-audit.json')
    assert sha(ROOT/'plan.json')==candidate['plan_sha256'] and sha(ROOT/'best.pt')==candidate['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in candidate['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in candidate['runtime_sources'].items())
    records=admit(plan,candidate,audit)
    data={s:dict(images=[],uv=[],meta=[]) for s in ('train','validation')};manifest=[];native_cal=None
    for spec,record,split in records:
        path=Path(record['path']);assert sha(path)==record['sha256']
        with h5py.File(path) as f:
            d=f['data/demo_0'];size=decode(d['obs/cup_detail_image'][0],'jpeg95').shape[0]
            k=f['calibration/cup_detail/intrinsics'][:].copy();k[:2]*=384/size
            cal=dict(K=k.tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[384,384])
            if native_cal is not None:assert cal==native_cal
            native_cal=cal;z=1.04+spec['declared_cup_geometry']['rim_z_m'];uv=project(np.r_[spec['cup_world'][:2],z],cal)
            indices=[('obs',i) for i in range(12)]+[('next_obs',-1)]
            for group,i in indices:
                image=decode(d[group+'/cup_detail_image'][i],'jpeg95')
                data[split]['images'].append(np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy())
                data[split]['uv'].append(uv)
                data[split]['meta'].append(dict(name=spec['name'],domain='native_detail',
                    baseline_cup=spec['cup_height_scale']==spec['cup_diameter_scale']==1.,
                    cup_height_scale=spec['cup_height_scale'],cup_diameter_scale=spec['cup_diameter_scale'],
                    camera=cal,z=z,xy=spec['cup_world'][:2],group=group,index=i))
        manifest.append(dict(name=spec['name'],split=split,path=str(path),sha256=record['sha256'],
                             cup_height_scale=spec['cup_height_scale'],cup_diameter_scale=spec['cup_diameter_scale']))
    for split,a in data.items():
        a['x']=torch.tensor(np.stack(a.pop('images')),device='cuda');a['u']=torch.tensor(np.array(a['uv']),device='cuda',dtype=torch.float32)
        a['weights']=torch.tensor(balanced_frame_weights(a['meta']),device='cuda')
    write('cup-locator-data-manifest.json',manifest)
    return plan,data,native_cal

@torch.inference_mode()
def validate(model,data):
    uv=[];score=[];absent=[]
    for start in range(0,len(data['x']),16):
        x=data['x'][start:start+16];u,s=centers(model(x));uv.extend(u.cpu().tolist());score.extend(s.cpu().tolist())
        negative=x.clone()
        for j,p in enumerate(data['u'][start:start+16].round().long().tolist()):
            x0,y0=p;negative[j,max(0,y0-24):min(384,y0+48),max(0,x0-50):min(384,x0+50)]=180
        _,n=centers(model(negative));absent.extend(n.cpu().tolist())
    rows=[]
    for m,u,s,n in zip(data['meta'],uv,score,absent):
        xy=plane_point(u,m['z'],m['camera'])[:2]
        rows.append(dict(name=m['name'],domain=m['domain'],baseline_cup=m['baseline_cup'],group=m['group'],index=m['index'],error_mm=float(np.linalg.norm(xy-m['xy'])*1000),score=s,synthetic_absent_score=n,uv=u))
    native=[r for r in rows if r['domain']=='native_detail'];primary=native
    minimum=min(r['score'] for r in primary);maximum=max(r['synthetic_absent_score'] for r in native)
    return dict(max_error_mm=max(r['error_mm'] for r in primary),mean_error_mm=float(np.mean([r['error_mm'] for r in primary])),
        all_size_max_error_mm=max(r['error_mm'] for r in native),minimum_positive_score=minimum,maximum_synthetic_absent_score=maximum,
        margin=minimum-maximum,threshold=(minimum+maximum)/2,rows=rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refine-from',type=Path);parser.add_argument('--member',type=int,choices=(0,1));args=parser.parse_args()
    refinement=None
    if args.refine_from is not None:
        refinement=(ROOT/args.refine_from).resolve();assert refinement.is_relative_to(ROOT) and refinement.is_file() and refinement.suffix=='.pt'
    candidate=read(ROOT/'candidate.json');assert candidate.get('seed_and_camera_accepted') and not candidate.get('trained')
    assert not (ROOT/'component-selection.json').exists(), 'Preserve a selected candidate'
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    assert not (ROOT/'cup-neural-localization.json').exists();torch.set_num_threads(4);torch.manual_seed(261109+(args.member or 0))
    start=time.perf_counter();plan,data,cal=collect(dense=refinement is not None);assert plan['primary_perception_scope']=='all_nine_declared_cup_sizes'
    train=data['train'];validation=data['validation'];model=CupLocator().cuda()
    parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    bank=parent['cup_bank'] if refinement is None else torch.load(refinement,map_location='cpu',weights_only=False)['bank']
    assert bank['kind'] in ('cup_heatmap','cup_heatmap_ensemble')
    member=(args.member or 0) if bank['kind']=='cup_heatmap_ensemble' else None
    if member is None:assert args.member is None
    state=bank['state'] if member is None else bank['members'][member]['state']
    # Retain the trained detector as initialization. Its old baseline-only
    # affine correction is not silently applied to all resized cups.
    model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
    yy,xx=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(96,device='cuda'),indexing='ij');best=None;best_attempt=None;history=[];steps=0
    fit_start=time.perf_counter()
    epochs=240
    for epoch in range(1,epochs+1):
        model.train()
        if epoch in (121,181):
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.multinomial(train['weights'],len(train['x']),replacement=True).split(16):
            x=train['x'][ids].float();u=train['u'][ids].clone();n=len(x)
            dx,dy=np.random.default_rng(261109+steps).integers(-28,29,2).tolist()
            padded=F.pad(x.permute(0,3,1,2),(32,32,32,32),value=180);x=padded[:,:,32-dy:416-dy,32-dx:416-dx].permute(0,2,3,1).contiguous();u+=u.new_tensor([dx,dy])
            positive=torch.rand(n,device='cuda')>.25
            for j in torch.nonzero(~positive,as_tuple=True)[0].tolist():
                px,py=u[j].round().long().tolist();x[j,max(0,py-24):min(384,py+48),max(0,px-50):min(384,px+50)]=180
            x=(x*(.95+.1*torch.rand(n,1,1,1,device='cuda'))).clamp(0,255)
            cell=(u/4).floor().long();assert (cell>=0).all() and (cell<96).all()
            gaussian=torch.exp(-((xx[None]-cell[:,0,None,None])**2+(yy[None]-cell[:,1,None,None])**2)/(2*1.5**2))*positive[:,None,None]
            output=model(x);logits=output[:,0];prob=logits.sigmoid();pos=gaussian==1;neg=~pos
            heat=(-(1-prob).square()*F.logsigmoid(logits)*pos-prob.square()*F.logsigmoid(-logits)*(1-gaussian)**4*neg).sum()/positive.sum().clamp(min=1)
            index=torch.arange(n,device='cuda');offset=output[index,1:,cell[:,1],cell[:,0]].sigmoid()
            offset_loss=F.smooth_l1_loss(offset[positive],(u/4-cell)[positive]) if positive.any() else output.sum()*0
            loss=heat+2*offset_loss;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        if epoch%10==0:
            model.eval();result=validate(model,validation);row=dict(epoch=epoch,**{k:v for k,v in result.items() if k!='rows'});history.append(row)
            if best_attempt is None or result['max_error_mm']<best_attempt['validation']['max_error_mm']:
                best_attempt=dict(epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},validation=result)
            training=validate(model,train)
            minimum=min(result['minimum_positive_score'],training['minimum_positive_score']);maximum=max(result['maximum_synthetic_absent_score'],training['maximum_synthetic_absent_score'])
            row.update(training_max_error_mm=training['max_error_mm'],joint_absence_margin=minimum-maximum)
            eligible=max(result['max_error_mm'],training['max_error_mm'])<plan['component_gates']['localization_max_error_mm'] and minimum-maximum>plan['component_gates']['localization_negative_margin']
            if eligible and (best is None or result['max_error_mm']<best['validation']['max_error_mm']):
                best=dict(epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},validation=result,training=training,threshold=(minimum+maximum)/2)
            if epoch%20==0:print(json.dumps(row),flush=True)
    report=dict(passed=best is not None,scope='all_nine_declared_cup_sizes',epochs=epochs,optimizer_steps=steps,learned_parameters=sum(p.numel() for p in model.parameters()),
        plan_sha256=sha(ROOT/'plan.json'),audit_sha256=sha(ROOT/'collect-r0-audit.json'),manual_review_sha256=sha(ROOT/'collect-r0-manual-review.json'),
        initialization=dict(path=str(ROOT/'best.pt') if refinement is None else str(refinement),sha256=sha(ROOT/'best.pt' if refinement is None else refinement),member=member),native_frames_per_episode=13,
        fit_seconds=time.perf_counter()-fit_start,total_seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),source_sha256=sha(Path(__file__)),module_sha256=sha(Path('/workspace/src/pour_cup_locator.py')),admission_helper_sha256=sha(ROOT/'scale_data_admission.py'),
        counts={s:dict(images=len(a['x']),episodes=len({m['name'] for m in a['meta']}),native_images=sum(m['domain']=='native_detail' for m in a['meta'])) for s,a in data.items()},
        history=history,selected=None if best is None else dict(epoch=best['epoch'],**best['validation']),
        best_attempt=dict(epoch=best_attempt['epoch'],**best_attempt['validation']),receptive_field_pixels=129,
        method='Fully convolutional RGB rim-center heatmap with subpixel offsets. Focal heatmap loss plus 2x smooth-L1 offsets, AdamW, integer image translation and photometric augmentation; generated absent-cup masks are training/validation diagnostics. Equal probability per size and per episode within each size. Whole-episode splits. All-size selection, including the baseline. No reserved images or simulator truth at inference.')
    torch.save(dict(state=best_attempt['state'],validation=best_attempt['validation'],epoch=best_attempt['epoch'],accepted=False),ROOT/'cup-neural-localization-attempt.pt')
    report['attempt_checkpoint_sha256']=sha(ROOT/'cup-neural-localization-attempt.pt')
    if best:
        model.load_state_dict(best['state']);model.eval();training=validate(model,train);report['training_check']=training
        report['passed']=training['max_error_mm']<plan['component_gates']['localization_max_error_mm']
        if report['passed']:
            identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(best['state'].items()))).hexdigest()
            bank=dict(kind='cup_heatmap',state=best['state'],identity=identity)
            threshold=best['threshold'];torch.save(dict(bank=bank,calibration=cal,threshold=threshold,
                consensus=dict(window=20,votes=8,radius_px=1.5,fraction=.6,score_floor=max(.01,threshold*.6))),ROOT/'cup-neural-localization.pt')
            report['checkpoint_sha256']=sha(ROOT/'cup-neural-localization.pt')
            write('scope-selection.json',dict(passed=True,scope='all_nine_declared_cup_sizes',epoch=best['epoch'],checkpoint_sha256=report['checkpoint_sha256'],
                max_validation_error_mm=best['validation']['max_error_mm'],max_training_error_mm=training['max_error_mm'],threshold=threshold,
                all_size_max_validation_error_mm=best['validation']['all_size_max_error_mm'],source_sha256=sha(Path(__file__)),
                rationale='Selection uses maximum error across every declared size in both whole-episode splits. Equal size/episode sampling. No unused holdout images or automatic runtime activation; level, independent holdout and native task gates remain pending.'))
    write('cup-neural-localization.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('history','selected','best_attempt','training_check','method')}),flush=True)
if __name__=='__main__':main()
