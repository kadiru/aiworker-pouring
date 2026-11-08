"""Separate supervised wrist-visible-bottle component; no active-model mutation."""
import copy,hashlib,json,sys,time,urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from pour_wide_empty import EmptyWrist,wrist_crop,rotation,augment
from validate_pour_recording import h5py,decode
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
WIDE=Path('/workspace/learning/A-wide-20261105-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def label(pose,T,K,roi):
    local=np.array([[x,y,z] for x in (-.025,.025) for y in (-.025,.025) for z in (.004,.135)])
    world=local@rotation(pose).T+pose[:3];v=(world-T[:3,3])@T[:3,:3]
    if (v[:,2]>=-.005).any():return None
    uv=np.stack([K[0,0]*v[:,0]/-v[:,2]+K[0,2],K[1,2]-K[1,1]*v[:,1]/-v[:,2]],1)
    lo=uv.min(0);hi=uv.max(0);rlo=np.asarray(roi[:2]);rhi=np.asarray(roi[2:])
    intersection=np.prod(np.maximum(0,np.minimum(hi,rhi)-np.maximum(lo,rlo)))
    if intersection==0:return 0.
    coverage=intersection/np.prod(hi-lo)
    return 1. if coverage>=.8 else None

def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    assert not (ROOT/'presence-component.json').exists()
    torch.set_num_threads(4);torch.manual_seed(261108)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);roi=saved['wide_empty_roi']
    manifest=read(WIDE/'empty-readout-candidates.json')['data_manifest']
    # New native empty-hand fixtures provide real absent-object negatives;
    # their independent validation layouts never enter fitting.
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*PV_A_presence_absent_*/episode.hdf5')):
        spec=read(p.parent/'parameters.json')
        manifest.append(dict(path=str(p),name=spec['name'],split=spec['study_split'].removeprefix('presence_'),
            sha256=sha(p),empty_allowance_ml=max(.25,.01*spec['initial_ml'])))
    data={s:dict(image=[],y=[],name=[],index=[],empty=[],tilt=[]) for s in ('train','validation')};records=[]
    start=time.perf_counter()
    for entry in manifest:
        p=Path(entry['path']);assert sha(p)==entry['sha256'];indices=[];excluded=0
        with h5py.File(p) as f:
            d=f['data/demo_0'];K=f['calibration/wrist/intrinsics'][:]
            for i in range(0,len(d['next_time']),5):
                y=label(d['ground_truth/bottle_pose'][i],d['next_obs/wrist_world_T_camera'][i],K,roi)
                if y is None:excluded+=1;continue
                a=data[entry['split']];a['image'].append(wrist_crop(decode(d['next_obs/wrist_image'][i],'jpeg95'),roi))
                a['y'].append(y);a['name'].append(entry['name']);a['index'].append(i)
                a['empty'].append(float(d['ground_truth/bottle_ml'][i])<=entry['empty_allowance_ml'])
                a['tilt'].append(float(d['ground_truth/bottle_tilt_deg'][i]));indices.append(i)
        records.append(dict(name=entry['name'],split=entry['split'],path=str(p),sha256=entry['sha256'],indices=indices,ambiguous_excluded=excluded))
    assert set(data['train']['name']).isdisjoint(data['validation']['name'])
    write('presence-data-audit.json',dict(counts={s:dict(images=len(a['y']),positive=sum(a['y']),negative=len(a['y'])-sum(a['y'])) for s,a in data.items()},
        manifest=records,scope='Projected presence label audit; no model fitting unless both splits have positive and negative images.'))
    for s,a in data.items():
        assert set(a['y'])=={0.,1.},s
        a['x']=torch.from_numpy(np.stack(a.pop('image'))).cuda();a['y']=torch.tensor(a['y'],device='cuda')
    tr,va=data['train'],data['validation'];counts=Counter(zip(tr['name'],tr['y'].tolist()))
    weight=tr['y'].new_tensor([1/counts[n,y] for n,y in zip(tr['name'],tr['y'].tolist())]);weight/=weight.mean()
    model=EmptyWrist().cuda();model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['wide_empty_model'].items() if k.startswith('encoder.')})
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);history=[];best=None;steps=0
    fit_start=time.perf_counter()
    for epoch in range(1,81):
        model.train()
        if epoch==51:
            for g in optimizer.param_groups:g['lr']/=3
        for ids in torch.randperm(len(tr['y']),device='cuda').split(128):
            optimizer.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(augment(tr['x'][ids])),tr['y'][ids],reduction='none')*weight[ids]).mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();steps+=1
        model.eval()
        with torch.no_grad():
            p=torch.cat([model(x).sigmoid() for x in va['x'].split(128)]);positive=va['y']>.5
            eligible=torch.tensor(np.asarray(va['tilt'])>=75,device='cuda')&positive
            empty=torch.tensor(va['empty'],device='cuda')&eligible
            row=dict(epoch=epoch,bce=float(torch.nn.functional.binary_cross_entropy(p,va['y'])),
                false_positive=int(((p>=.7)&~positive).sum()),missed_positive=int(((p<.7)&positive).sum()),
                held_tilt_min_probability=float(p[eligible].min()),empty_tilt_min_probability=float(p[empty].min()),
                positive_recall=float((p[positive]>=.7).float().mean()))
            history.append(row)
            if row['false_positive']==0 and row['held_tilt_min_probability']>=.7 and row['positive_recall']>=.99 and (best is None or row['bce']<best['row']['bce']):
                best=dict(row=copy.deepcopy(row),state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if epoch%10==0:print(json.dumps(row),flush=True)
    report=dict(method='Supervised binary cross entropy, separate wrist CNN initialized from the existing empty-bottle encoder; AdamW, episode/class weighting, image augmentation.',
        device=torch.cuda.get_device_name(),fit_seconds=time.perf_counter()-fit_start,total_seconds=time.perf_counter()-start,
        learned_parameters=sum(p.numel() for p in model.parameters()),epochs=80,optimizer_steps=steps,
        counts={s:dict(episodes=len(set(a['name'])),images=len(a['y']),positive=int(a['y'].sum()),negative=int((1-a['y']).sum())) for s,a in data.items()},
        passed=best is not None,selected=None if best is None else best['row'],roi=roi,data_manifest=records,candidates=history,
        labels='Offline projected bottle envelope: >=80% within fixed wrist ROI positive, zero overlap negative, ambiguous excluded. This labels projected bottle presence, not guaranteed optical visibility; live control requires previously verified grasp and measured finger obstruction.',
        gate='Validation: zero false positives at p>=0.7, every eligible tilted positive >=0.7, total positive recall >=99%. No reserved test frames.',
        active_checkpoint_unchanged=True,source_sha256=sha(Path(__file__)))
    if best:
        torch.save(dict(state=best['state'],roi=roi),ROOT/'presence-component.pt');report['checkpoint_sha256']=sha(ROOT/'presence-component.pt')
    write('presence-component.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('data_manifest','candidates')}))

if __name__=='__main__':main()
