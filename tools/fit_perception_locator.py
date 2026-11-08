"""4070 supervised point localization using whole-episode train/validation splits."""
import copy,hashlib,json,sys,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import project,plane_point
from pour_cup_locator import CupLocator,centers
from perception_fit_data import admitted_scenes
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def collect():
    plan=read(ROOT/'plan.json');audit=read(ROOT/'recording-audit.json');indexed={r['name']:r for r in audit['records']}
    data={s:dict(images=[],uv=[],meta=[]) for s in ('train','validation')};manifest=[];native_cal=None
    def add(path,spec,split,camera,domain,indices):
        nonlocal native_cal
        with h5py.File(path) as f:
            d=f['data/demo_0'];size=decode(d['obs/'+camera+'_image'][0],'jpeg95').shape[0]
            k=f['calibration/'+camera+'/intrinsics'][:].copy();k[:2]*=384/size
            cal=dict(K=k.tolist(),T=d['obs/'+camera+'_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[384,384])
            g=spec.get('declared_cup_geometry',dict(rim_z_m=.038));z=1.04+g['rim_z_m'];uv=project(np.r_[spec['cup_world'][:2],z],cal)
            if domain=='native_detail':native_cal=cal
            for group,i in indices:
                image=np.asarray(Image.fromarray(decode(d[group+'/'+camera+'_image'][i],'jpeg95')).resize((384,384),Image.Resampling.BILINEAR)).copy()
                data[split]['images'].append(image);data[split]['uv'].append(uv)
                data[split]['meta'].append(dict(name=spec['name'],domain=domain,baseline_cup=spec.get('cup_height_scale',1.)==spec.get('cup_diameter_scale',1.)==1.,camera=cal,z=z,xy=spec['cup_world'][:2],group=group,index=i))
    for s,split,admission in admitted_scenes(plan):
        r=indexed[s['name']];path=Path(r['path']);assert sha(path)==r['sha256']
        add(path,s,split,'cup_detail','native_detail',[('obs',0),('obs',11),('next_obs',-1)])
        manifest.append(dict(name=s['name'],split=split,recorded_split=s['study_split'],admission=admission,path=str(path),sha256=r['sha256'],domain='native_detail'))
    for r in read(Path('/workspace/learning/A-wide-20261105-v1/empty-readout-candidates.json'))['data_manifest']:
        path=Path(r['path']);assert sha(path)==r['sha256'];s=read(path.parent/'parameters.json')
        add(path,s,r['split'],'task','historical_task_replay',[('obs',0),('next_obs',-1)])
        manifest.append(dict(name=s['name'],split=r['split'],path=str(path),sha256=r['sha256'],domain='historical_task_replay'))
    assert {r['name'] for r in manifest if r['split']=='train'}.isdisjoint({r['name'] for r in manifest if r['split']=='validation'})
    for s,a in data.items():a['x']=torch.tensor(np.stack(a.pop('images')),device='cuda');a['u']=torch.tensor(np.array(a['uv']),device='cuda',dtype=torch.float32)
    write('cup-locator-data-manifest.json',manifest);return plan,data,native_cal

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
    native=[r for r in rows if r['domain']=='native_detail'];primary=[r for r in native if r['baseline_cup']]
    minimum=min(r['score'] for r in primary);maximum=max(r['synthetic_absent_score'] for r in native)
    return dict(max_error_mm=max(r['error_mm'] for r in primary),mean_error_mm=float(np.mean([r['error_mm'] for r in primary])),
        all_size_max_error_mm=max(r['error_mm'] for r in native),minimum_positive_score=minimum,maximum_synthetic_absent_score=maximum,
        margin=minimum-maximum,threshold=(minimum+maximum)/2,rows=rows)

def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    assert not (ROOT/'cup-neural-localization.json').exists();torch.set_num_threads(4);torch.manual_seed(261109)
    start=time.perf_counter();plan,data,cal=collect();assert plan['primary_perception_scope']=='baseline_cup_only'
    train=data['train'];validation=data['validation'];model=CupLocator().cuda()
    parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    state=model.state_dict()
    for i in (0,2,4):
        for suffix in ('weight','bias'):state[f'net.{i}.{suffix}']=parent['cup_model'][f'encoder.net.{i}.{suffix}']
    model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    yy,xx=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(96,device='cuda'),indexing='ij');best=None;best_attempt=None;history=[];steps=0
    fit_start=time.perf_counter()
    for epoch in range(1,601):
        model.train()
        if epoch in (301,451):
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.randperm(len(train['x']),device='cuda').split(16):
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
            eligible=max(result['max_error_mm'],training['max_error_mm'])<plan['component_gates']['localization_max_error_mm'] and minimum-maximum>plan['component_gates']['localization_negative_margin']
            if eligible and (best is None or result['max_error_mm']<best['validation']['max_error_mm']):
                best=dict(epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},validation=result,training=training,threshold=(minimum+maximum)/2)
            if epoch%20==0:print(json.dumps(row),flush=True)
    report=dict(passed=best is not None,scope='baseline_cup_only',epochs=600,optimizer_steps=steps,learned_parameters=sum(p.numel() for p in model.parameters()),
        fit_seconds=time.perf_counter()-fit_start,total_seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),source_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'staged/pour_cup_locator.py'),admission_helper_sha256=sha(ROOT/'perception_fit_data.py'),
        counts={s:dict(images=len(a['x']),episodes=len({m['name'] for m in a['meta']}),native_images=sum(m['domain']=='native_detail' for m in a['meta'])) for s,a in data.items()},
        history=history,selected=None if best is None else dict(epoch=best['epoch'],**best['validation']),
        best_attempt=dict(epoch=best_attempt['epoch'],**best_attempt['validation']),receptive_field_pixels=129,
        method='Fully convolutional RGB rim-center heatmap with subpixel offsets. Focal heatmap loss plus 2x smooth-L1 offsets, AdamW, integer image translation and photometric augmentation; generated absent-cup masks are training/validation diagnostics. Whole-episode splits. No reserved images or simulator truth at inference.')
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
            write('scope-selection.json',dict(passed=True,scope='baseline_cup_only',epoch=best['epoch'],checkpoint_sha256=report['checkpoint_sha256'],
                max_validation_error_mm=best['validation']['max_error_mm'],max_training_error_mm=training['max_error_mm'],threshold=threshold,
                all_size_max_validation_error_mm=best['validation']['all_size_max_error_mm'],source_sha256=sha(Path(__file__)),
                rationale='Baseline scope declared before this refit; added only training scenes and explicitly admitted calibration pilots. Resized autonomy remains gated; all-size errors retained. No reserved data used.'))
    write('cup-neural-localization.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('history','selected','best_attempt','training_check','method')}),flush=True)
if __name__=='__main__':main()
