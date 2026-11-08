"""4070 supervised cup localization and level fit from disjoint native scenes.

Writes component artifacts only. No active checkpoint replacement, test-data
access, or changes to physics/task scores occur in this helper.
"""
import copy,hashlib,json,sys,time,urllib.request
from itertools import product
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import make_bank,locate,project,plane_point,cup_crop,surface_features
from pour_guard_models import CupEstimator
from pour_perception_geometry import settled_label
from pour_temporal_localization import consensus
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def collect():
    plan=read(ROOT/'plan.json');expected={s['name']:s for s in plan['collect']}
    seq=[];manifest=[];cal=None
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')):
        spec=read(path)
        if spec['name'] not in expected:continue
        assert spec['study_split'] in ('train','validation') and all(spec[k]==v for k,v in expected[spec['name']].items())
        h5=path.parent/'episode.hdf5';assert h5.exists();rows=read(path.parent/'samples.json');label=settled_label(rows)
        entry=dict(name=spec['name'],split=spec['study_split'],path=str(h5),sha256=sha(h5),label=label)
        manifest.append(entry)
        with h5py.File(h5) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0'];current=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[1024,1024])
            if cal is None:cal=current
            else:assert np.allclose(cal['K'],current['K']) and np.allclose(cal['T'],current['T'])
            images=[np.asarray(Image.fromarray(decode(d['obs/cup_detail_image'][i],'jpeg95')).resize((384,384),Image.Resampling.BILINEAR)).copy() for i in range(12)]
        seq.append(dict(spec=spec,entry=entry,images=images))
    assert len(seq)==len(expected) and len({e['spec']['name'] for e in seq})==len(expected)
    write('cup-data-manifest.json',manifest)
    return plan,seq,cal

@torch.inference_mode()
def localize(seq,detail,gate):
    cal=copy.deepcopy(detail);K=np.array(cal['K']);K[:2]*=384/1024;cal['K']=K.tolist();cal['resolution_wh']=[384,384]
    train=[e for e in seq if e['spec']['study_split']=='train'];anchors=[]
    for e in train:anchors.append(project(np.array(e['spec']['cup_world'])+[0,0,e['spec']['declared_cup_geometry']['rim_z_m']],cal))
    xy=np.array([e['spec']['cup_world'][:2] for e in train]);lo=xy.min(0)-.025;hi=xy.max(0)+.025
    region=[project([x,y,1.078],cal).tolist() for x,y in ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))]
    candidates=[]
    for absolute,clip,mask in product((False,True),(.05,.1,None),((10,0,70,16),(10,2,70,24),(16,4,64,22),None)):
        bank=make_bank([e['images'][0] for e in train],anchors,'cup',aggregation='max',absolute_gradients=absolute,
            gradient_clip=clip,cup_mask_box=mask,anchor_radius_px=48.)
        bank['bounds']=[24,24,360,356];bank['initial_region_uv']=region;rows=[]
        for e in seq:
            image=e['images'][0];s=e['spec'];z=1.04+s['declared_cup_geometry']['rim_z_m'];uv,score=locate(image,bank)
            point=plane_point(uv,z,cal)[:2];error=float(np.linalg.norm(point-np.array(s['cup_world'][:2]))*1000)
            true=project(np.array(s['cup_world'])+[0,0,z-1.04],cal);x,y=np.rint(true).astype(int)
            negative=image.copy();negative[max(0,y-20):min(384,y+35),max(0,x-45):min(384,x+45)]=180
            _,negative_score=locate(negative,bank)
            rows.append(dict(name=s['name'],split=s['study_split'],error_mm=error,score=score,synthetic_absent_score=negative_score))
        minimum=min(r['score'] for r in rows);maximum=max(r['synthetic_absent_score'] for r in rows)
        row=dict(absolute_gradients=absolute,gradient_clip=clip,cup_mask_box=mask,max_error_mm=max(r['error_mm'] for r in rows),
            margin=minimum-maximum,threshold=(minimum+maximum)/2,rows=rows)
        candidates.append((row,bank));print('CUP_LOCALIZATION',json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
    write('cup-localization-candidates.json',[r for r,b in candidates])
    eligible=[(r,b) for r,b in candidates if r['max_error_mm']<gate['localization_max_error_mm'] and r['margin']>gate['localization_negative_margin']]
    if not eligible:return None,dict(passed=False,reason='No precise, occlusion-rejecting bank',best=min((r for r,b in candidates),key=lambda r:r['max_error_mm']))
    chosen,bank=min(eligible,key=lambda rb:(rb[0]['max_error_mm'],-rb[0]['margin']))
    config=dict(window=20,votes=8,radius_px=1.5,fraction=.6,score_floor=max(.25,chosen['threshold']-.2));windows=[]
    for e in seq:
        history=[];accepted=None;s=e['spec'];z=1.04+s['declared_cup_geometry']['rim_z_m']
        for image in e['images']:
            uv,score=locate(image,bank);history.append((uv.tolist(),score));accepted=consensus(history,chosen['threshold'],config)
            if accepted:break
        if accepted is None:return None,dict(passed=False,reason='Temporal localization unconfirmed',name=s['name'],chosen=chosen)
        xy=plane_point(accepted['uv'],z,cal)[:2];error=float(np.linalg.norm(xy-s['cup_world'][:2])*1000)
        if error>=gate['localization_max_error_mm']:return None,dict(passed=False,reason='Temporal localization inaccurate',name=s['name'],error_mm=error)
        e['localized_xy']=xy;windows.append(dict(name=s['name'],error_mm=error,frames=len(history)))
    return dict(bank=bank,calibration=cal,threshold=chosen['threshold'],consensus=config),dict(passed=True,chosen=chosen,windows=windows,
        method='Supervised gradient-template banks and confidence/temporal-consensus calibration; fixture dimensions are declared, not visually inferred. Synthetic absent masks are a diagnostic, not an independent real missing-cup test.')

def fit_level(seq,cal,gate,parent):
    data={s:dict(image=[],target=[],name=[],domain=[]) for s in ('train','validation')}
    for e in seq:
        label=e['entry']['label'];s=e['spec']
        if not label['accepted']:continue
        actual=copy.deepcopy(cal);actual['cup_geometry']=s['declared_cup_geometry'];a=data[s['study_split']]
        with h5py.File(e['entry']['path']) as f:
            d=f['data/demo_0'];times=d['next_time'][:];ids=np.flatnonzero(times>=times[-1]-1.-1e-7)
            for i in ids[::2]:
                image=cup_crop(decode(d['next_obs/cup_detail_image'][i],'jpeg95'),e['localized_xy'],actual)
                a['image'].append(image);a['target'].append(label['target']);a['name'].append(s['name']);a['domain'].append('native_detail')
    # Rehearse earlier real pouring appearances to avoid replacing dynamic
    # perception with a model trained only on stationary high-resolution cups.
    # These remain original lower-resolution task images, never synthetic
    # detail-camera recordings. Only established train/validation references
    # are read; no historical or new reserved test images enter this fit.
    replay=[]
    for entry in read(Path('/workspace/learning/A-wide-20261105-v1/empty-readout-candidates.json'))['data_manifest']:
        path=Path(entry['path']);assert sha(path)==entry['sha256'];rows=read(path.parent/'samples.json');used=[]
        with h5py.File(path) as f:
            d=f['data/demo_0'];uv,score=locate(decode(d['obs/task_image'][0],'jpeg95'),parent['cup_bank'])
            assert score>=parent['thresholds']['cup_localization_score']
            xy=plane_point(uv,1.078,parent['calibration'])[:2];a=data[entry['split']]
            for i in np.unique(np.linspace(0,len(rows)-1,24,dtype=int)):
                r=rows[i]
                if r['ever_table_ml'] or r['spilled_ml']:continue
                target=r['cup_fill_fraction'];label_source='instantaneous_dynamic_surface'
                if i>=20:
                    window=rows[:i+1];stable=settled_label(window)
                    if stable['accepted']:target=stable['target'];label_source='measured_trailing_stable_median'
                image=cup_crop(decode(d['next_obs/task_image'][int(i)],'jpeg95'),xy,parent['calibration'])
                a['image'].append(image);a['target'].append(target);a['name'].append(entry['name']);a['domain'].append('historical_task_replay')
                used.append(dict(index=int(i),target=target,label_source=label_source))
        replay.append(dict(name=entry['name'],split=entry['split'],path=str(path),sha256=entry['sha256'],frames=used))
    write('cup-replay-manifest.json',replay)
    assert set(data['train']['name']).isdisjoint(data['validation']['name'])
    for s,a in data.items():
        assert len({n for n,d in zip(a['name'],a['domain']) if d=='native_detail'})>=(20 if s=='train' else 14),f'Insufficient stable episodes in {s}'
        a['x']=torch.from_numpy(np.stack(a.pop('image'))).cuda();a['y']=torch.tensor(a['target'],device='cuda')
    tr,va=data['train'],data['validation'];native=torch.tensor([d=='native_detail' for d in va['domain']],device='cuda')
    near=(va['y']>.85)&native
    assert int(near.sum())>=20 and float(va['y'][native].max())>=.94,'No near-full validation coverage'
    model=CupEstimator().cuda();model.load_state_dict(parent['cup_model']);model.eval()
    with torch.no_grad():
        baseline=torch.cat([model(x) for x in va['x'].split(128)])+parent['cup_bias']
        replay_baseline_mae=float((baseline.clamp(0,1)-va['y'])[~native].abs().mean())
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    domain_weight=torch.tensor([4. if d=='native_detail' else 1. for d in tr['domain']],device='cuda')
    history=[];best=None;steps=0
    for epoch in range(1,241):
        model.train()
        if epoch in (121,181):
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.randperm(len(tr['y']),device='cuda').split(64):
            x=tr['x'][ids].float();x=(x*(.98+.04*torch.rand(len(x),1,1,1,device='cuda'))).clamp(0,255)
            opt.zero_grad();error=model(x)-tr['y'][ids];loss=(error.square()*torch.where(tr['y'][ids]>.8,3.,1.)*domain_weight[ids]).mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        model.eval()
        with torch.no_grad():
            pred=torch.cat([model(x) for x in va['x'].split(128)]);error=pred-va['y'];row=dict(epoch=epoch,
                mae=float(error[native].abs().mean()),near_mae=float(error[near].abs().mean()),near_max=float(error[near].abs().max()),near_overestimate=float(error[near].max()),
                historical_replay_mae=float(error[~native].abs().mean()))
            history.append(row);score=row['mae']+row['near_mae']+.3*row['near_max']
            eligible=row['mae']<gate['level_validation_mae'] and row['near_max']<gate['level_validation_near_max_error'] and row['near_overestimate']<gate['level_validation_near_overestimate'] and row['historical_replay_mae']<=max(.05,replay_baseline_mae+.015)
            if eligible and (best is None or score<best['score']):best=dict(row=row,score=score,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if epoch%20==0:print('CUP_LEVEL',json.dumps(row),flush=True)
    with torch.no_grad():
        train_native=torch.tensor([d=='native_detail' for d in tr['domain']],device='cuda')
        x=surface_features(tr['x'][train_native])[0].double();vx=surface_features(va['x'][native])[0].double();y=tr['y'][train_native].double();geo=[]
        for ridge in (1e-6,1e-4,.01,.1):
            reg=torch.eye(4,device='cuda',dtype=torch.float64)*ridge;reg[0,0]=1e-12
            coef=torch.linalg.solve(x.T@x+reg,x.T@y);err=(vx@coef).clamp(0,1)-va['y'][native]
            geo.append(dict(coef=coef.cpu().tolist(),dimensions=4,bias=0.,ridge=ridge,mae=float(err.abs().mean()),max_error=float(err.abs().max())))
        geometry=min(geo,key=lambda g:g['max_error'])
    passed=best is not None and geometry['max_error']<gate['geometry_validation_max_error']
    report=dict(passed=passed,selected=None if best is None else best['row'],geometry={k:v for k,v in geometry.items() if k!='coef'},
        counts={s:dict(episodes=len(set(a['name'])),images=len(a['y']),min_target=float(a['y'].min()),max_target=float(a['y'].max()),
            domains={d:dict(images=sum(v==d for v in a['domain']),episodes=len({n for n,v in zip(a['name'],a['domain']) if v==d})) for d in set(a['domain'])}) for s,a in data.items()},
        historical_replay_baseline_mae=replay_baseline_mae,replay_scope='Original lower-resolution task views from 24/8 train/validation references. New-record weight 4, replay weight 1. Dynamic appearances retained; no reserved tests or generated detail pixels.',
        candidates=history,learned_parameters=sum(p.numel() for p in model.parameters())+4,epochs=240,optimizer_steps=steps,
        method='Supervised RGB cup-level regression; pretrained CNN fine-tuned with weighted squared error, AdamW and mild photometric augmentation. Separate four-coefficient surface-geometry least squares. Episode-disjoint validation selects epoch; no reserved test images.',
        target_source='Measured final-second surface median for each stationary native recording; image-localized crops, declared cup dimensions. No inventory-derived target lookup.')
    artifact=None if not passed else dict(cup_model=best['state'],cup_bias=0.,surface_geometry=geometry,
        settled_regressor=dict(columns=[4],mean=[0.],scale=[1.],coef=[1.],output_bias=0.,method='Identity of fitted settled-level CNN output; no additional learned parameters.'))
    return artifact,report

def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    assert not (ROOT/'cup-component.json').exists();torch.set_num_threads(4);torch.manual_seed(261108)
    start=time.perf_counter();plan,seq,cal=collect();parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    loc,loc_report=localize(seq,cal,plan['component_gates']);write('cup-localization-validation.json',loc_report)
    report=dict(passed=False,device=torch.cuda.get_device_name(),localization_passed=loc is not None,source_sha256=sha(Path(__file__)))
    if loc is not None:
        level,level_report=fit_level(seq,cal,plan['component_gates'],parent);write('cup-level-validation.json',level_report)
        report.update(level_passed=level is not None)
        if level is not None:
            torch.save(dict(localization=loc,level=level,detail_calibration=cal),ROOT/'cup-component.pt')
            report.update(passed=True,checkpoint_sha256=sha(ROOT/'cup-component.pt'))
    report['total_seconds']=time.perf_counter()-start;write('cup-component.json',report);print(json.dumps(report))
if __name__=='__main__':main()
