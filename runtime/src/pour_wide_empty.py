"""Supervised high-resolution wrist classification, separate from old models."""
import copy,json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from pour_stop_models import VisualEncoder
from validate_pour_recording import h5py,decode


class EmptyWrist(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=VisualEncoder();self.head=torch.nn.Linear(64,1)
    def forward(self,image):return self.head(self.encoder(image))[:,0]


def wrist_crop(image,roi):
    return np.asarray(Image.fromarray(image).crop(tuple(roi)).resize((128,128),Image.Resampling.BILINEAR)).copy()


def load_empty(saved,device='cuda'):
    model=EmptyWrist().to(device).eval();model.load_state_dict(saved['wide_empty_model']);return model


def rotation(pose):
    x,y,z,w=pose[3:]/np.linalg.norm(pose[3:])
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def calibration(seq):
    points=[];names=[]
    # Labels learn one constant image crop, never live object state in control.
    local=np.array([[x,y,z] for x in (-.03,.03) for y in (-.03,.03) for z in (0.,.135)])
    for e in seq:
        if e['split']!='train':continue
        names.append(e['name'])
        with h5py.File(e['path']) as f:
            d=f['data/demo_0'];K=f['calibration/wrist/intrinsics'][:]
            ids=np.flatnonzero((d['ground_truth/bottle_tilt_deg'][:]>=75)&d['ground_truth/grasp_verified'][:].astype(bool))[::2]
            for i in ids:
                pose=d['ground_truth/bottle_pose'][i];T=d['next_obs/wrist_world_T_camera'][i]
                v=(local@rotation(pose).T+pose[:3]-T[:3,3])@T[:3,:3];assert np.all(v[:,2]<0)
                points.extend(np.stack([K[0,0]*v[:,0]/-v[:,2]+K[0,2],K[1,2]-K[1,1]*v[:,1]/-v[:,2]],1).tolist())
    a=np.asarray(points);lo=np.maximum(0,np.floor(a.min(0)-8)).astype(int);hi=np.minimum(384,np.ceil(a.max(0)+8)).astype(int)
    roi=[*lo.tolist(),*hi.tolist()];assert 32<hi[0]-lo[0]<160 and 32<hi[1]-lo[1]<180
    return dict(roi_xyxy=roi,training_names=names,envelope_radius_m=.03,envelope_height_m=.135,padding_px=8,
        method='Fixed wrist ROI learned from training-only projected bottle envelopes; no live poses or validation/test ROI fitting.')


def augment(image):
    x=image.float().permute(0,3,1,2);n=len(x);angle=(torch.rand(n,device=x.device)*2-1)*np.deg2rad(3.)
    scale=.95+.1*torch.rand(n,device=x.device);a=torch.zeros(n,2,3,device=x.device)
    a[:,0,0]=a[:,1,1]=angle.cos()*scale;a[:,0,1]=-angle.sin()*scale;a[:,1,0]=angle.sin()*scale
    a[:,:,2]=(torch.rand(n,2,device=x.device)*2-1)*(3./64.)
    grid=torch.nn.functional.affine_grid(a,x.shape,align_corners=False)
    y=torch.nn.functional.grid_sample(x,grid,padding_mode='border',align_corners=False)
    y=(y*(.9+.2*torch.rand(n,1,1,1,device=x.device))+(torch.rand(n,1,1,1,device=x.device)*2-1)*7.65).clamp(0,255)
    return torch.where(torch.rand(n,1,1,1,device=x.device)<.2,x,y).permute(0,2,3,1)


def statistics(probability,data,normal,slow):
    positive=data['y']>.5;negative=~positive;eligible=data['eligible'];episodes=[]
    for name in sorted(set(data['name'])):
        ids=[i for i,n in enumerate(data['name']) if n==name and positive[i] and eligible[i]]
        if ids:
            p=probability[ids];episodes.append(dict(name=name,positive_frames=len(ids),detected=int((p>=normal).sum()),min_probability=float(p.min()),max_probability=float(p.max())))
    metrics=dict(validation_bce=float(torch.nn.functional.binary_cross_entropy(probability,data['y'])),positive_frames=int(positive.sum()),negative_frames=int(negative.sum()),
        false_positive_normal=int(((probability>=normal)&negative&eligible).sum()),false_positive_slow=int(((probability>=slow)&negative&eligible).sum()),
        false_positive_slow_all_tilts=int(((probability>=slow)&negative).sum()),positive_detected=int(((probability>=normal)&positive&eligible).sum()))
    false_ids=torch.nonzero((probability>=slow)&negative&eligible).flatten().tolist()
    metrics['false_positive_samples']=[dict(name=data['name'][i],index=data['index'][i],source_ml=data['source_ml'][i],probability=float(probability[i])) for i in false_ids]
    metrics['maximum_eligible_negative_probability']=float(probability[negative&eligible].max())
    return metrics,episodes


def fit_readout(saved,seq,localized):
    from pour_wide_study import ROOT
    crop=calibration(seq);roi=crop['roi_xyxy'];arrays={s:dict(image=[],y=[],name=[],index=[],source_ml=[],eligible=[]) for s in ('train','validation')};manifest=[]
    (ROOT/'empty-roi-calibration.json').write_text(json.dumps(crop,indent=2)+'\n')
    for e in seq:
        path=Path(e['path']);rows=json.loads((path.parent/'samples.json').read_text());used=[]
        with h5py.File(path) as f:
            spec=json.loads(f.attrs['spec']);d=f['data/demo_0'];allowance=max(.25,.01*spec['initial_ml'])
            for i,r in enumerate(rows):
                if i%2 or r['bottle_tilt_deg']<75 or not r['grasp_verified'] or not(r['bottle_ml']<=allowance or r['bottle_ml']>=.5):continue
                a=arrays[e['split']];a['image'].append(wrist_crop(decode(d['next_obs/wrist_image'][i],'jpeg95'),roi))
                a['y'].append(float(r['bottle_ml']<=allowance));a['eligible'].append(r['bottle_tilt_deg']>=95 and r['source_outflow_ml_s']<3.)
                a['name'].append(e['name']);a['index'].append(i);a['source_ml'].append(r['bottle_ml']);used.append(i)
        manifest.append(dict(name=e['name'],split=e['split'],path=e['path'],sha256=e['sha256'],indices=used,empty_allowance_ml=allowance))
    with torch.inference_mode(False),torch.enable_grad():
        torch.manual_seed(20261105)
        data={s:dict(image=torch.from_numpy(np.stack(a['image'])).cuda().clone(),y=torch.tensor(a['y'],device='cuda'),eligible=torch.tensor(a['eligible'],device='cuda'),name=a['name'],index=a['index'],source_ml=a['source_ml']) for s,a in arrays.items()}
        tr,va=data['train'],data['validation'];assert set(tr['name']).isdisjoint(va['name'])
        counts=Counter(zip(tr['name'],arrays['train']['y']));weights=tr['y'].new_tensor([1./counts[n,y] for n,y in zip(tr['name'],arrays['train']['y'])]);weights/=weights.mean()
        model=EmptyWrist().cuda();model.encoder.load_state_dict({k.removeprefix('wrist.'):v for k,v in saved['source_detector'].items() if k.startswith('wrist.')})
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);normal=saved['thresholds']['empty_probability'];slow=saved['thresholds']['slow_empty_probability']
        history=[];best=None;steps=0
        for epoch in range(1,101):
            if epoch in (46,76):
                for group in optimizer.param_groups:group['lr']/=3.
            model.train();order=torch.randperm(len(tr['y']),device='cuda');total=0.
            for ids in order.split(128):
                optimizer.zero_grad();logit=model(augment(tr['image'][ids]))
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,tr['y'][ids],reduction='none')*weights[ids]).mean()
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();steps+=1;total+=float(loss)*len(ids)
            model.eval()
            with torch.no_grad():
                probability=torch.cat([model(x).sigmoid() for x in va['image'].split(128)]);metrics,episodes=statistics(probability,va,normal,slow)
                row=dict(epoch=epoch,steps=steps,train_loss=total/len(tr['y']),metrics=metrics,positive_episodes=episodes);history.append(row)
                eligible=metrics['false_positive_slow']==0 and episodes and all(e['detected']>=6 for e in episodes)
                if eligible and (best is None or metrics['validation_bce']<best['row']['metrics']['validation_bce']):best=dict(row=copy.deepcopy(row),state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            if epoch%5==0:print('EMPTY_WRIST_EPOCH',epoch,'bce',metrics['validation_bce'],'fp',metrics['false_positive_slow'],'positive_votes',[e['detected'] for e in episodes],flush=True)
        report=dict(method='Supervised binary classification: separate wrist CNN initialized from the parent wrist encoder and fine-tuned on a high-resolution fixed bottle crop, with episode/class-balanced BCE, AdamW and augmentation.',
            validation_operating_gate='Labeled tilt>=95 degrees and source outflow<3 nominal mL/s, an offline proxy for runtime proprioception/visual flow; all-angle errors reported. No test data.',
            learned_parameters=sum(p.numel() for p in model.parameters()),train_episodes=len(set(tr['name'])),validation_episodes=len(set(va['name'])),train_frames=len(tr['y']),validation_frames=len(va['y']),epochs=100,optimizer_steps=steps,roi_calibration=crop,
            optimizer=dict(name='AdamW',initial_lr=3e-4,weight_decay=1e-4,divide_lr_by_three_at_epochs=[46,76],batch_size=128),
            training_augmentation=dict(online=True,max_translation_px=3,max_rotation_deg=3,scale_range=[.95,1.05],gain_range=[.9,1.1],max_offset_fraction=.03,identity_probability=.2,seed=20261105,scope='Training crops only; validation and recordings remain native.'),
            existing_detector_weights_unchanged=True,candidates=history,data_manifest=manifest)
        (ROOT/'empty-readout-candidates.json').write_text(json.dumps(report,indent=2)+'\n')
        assert best is not None,'Wrist CNN has false-positive or missing-positive validation episodes; improve training before native use'
        saved['wide_empty_model']=best['state'];saved['wide_empty_roi']=roi
        return dict(**{k:v for k,v in report.items() if k not in ('candidates','data_manifest')},selected=best['row'])
