"""Static nine-size RGB level calibration; writes a separate candidate only.

No held-out images, active model replacement, dynamic-task claims or threshold
changes. Cup dimensions are declared inputs to the image crop.
"""
import copy
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_models import CupEstimator
from pour_guard_vision import locate,plane_point,cup_crop,color_geometry_readout
from pour_settled_model import features,predict
from pour_temporal_localization import consensus
from scale_data_admission import admit,balanced_frame_weights,SIZES,collection_paths
from scale_rim_refiner import locate_static
ROOT=Path('/workspace/learning/A-scale-20261109-v1')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n')


@torch.no_grad()
def collect(surface=False):
    plan=read(ROOT/'plan.json');m=read(ROOT/'candidate.json');audit_path,review_path=collection_paths(ROOT,m)
    audit=read(audit_path);review=read(review_path);selection=read(ROOT/'scope-selection.json')
    assert review['accepted'] and review['audit_sha256']==sha(audit_path)
    assert selection['passed'] and selection['scope']=='all_nine_declared_cup_sizes'
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert sha(ROOT/'cup-neural-localization.pt')==selection['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    loc=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False)
    resolution=loc['calibration']['resolution_wh'];assert resolution in ([384,384],[768,768],[1024,1024])
    if loc['bank']['kind']=='native_rim_refiner':assert selection['module_sha256']==sha(ROOT/'scale_rim_refiner.py')
    parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    model=CupEstimator().cuda().eval();model.load_state_dict(parent['cup_model'])
    data={s:dict(images=[],target=[],xy=[],meta=[]) for s in ('train','validation')};windows=[]
    if surface:
        from scale_surface_geometry import surface_crop,surface_lookup
        for a in data.values():a.update(surface_images=[],surface_lookup=[])
    for spec,r,split in admit(plan,m,audit):
        path=Path(r['path']);assert sha(path)==r['sha256']
        with h5py.File(path) as f:
            d=f['data/demo_0'];detail=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,
                resolution_wh=[1024,1024],cup_geometry=spec['declared_cup_geometry'])
            scaled=np.asarray(detail['K']).copy();scaled[:2]*=resolution[0]/1024
            assert np.allclose(scaled,loc['calibration']['K']) and np.allclose(detail['T'],loc['calibration']['T'])
            history=[];accepted=None
            for i in range(12):
                rgb=decode(d['obs/cup_detail_image'][i],'jpeg95')
                image=np.asarray(Image.fromarray(rgb).resize(tuple(resolution),Image.Resampling.BILINEAR)).copy()
                uv,score=locate_static(image,loc['bank']);history.append((uv.tolist(),score))
                accepted=consensus(history,loc['threshold'],loc['consensus'])
                if accepted is not None:break
            xy=None if accepted is None else plane_point(accepted['uv'],1.04+spec['declared_cup_geometry']['rim_z_m'],loc['calibration'])[:2]
            error=None if xy is None else float(np.linalg.norm(xy-spec['cup_world'][:2])*1000)
            windows.append(dict(name=spec['name'],confirmed=accepted is not None,error_mm=error,frames=len(history)))
            if xy is None or error>=plan['component_gates']['localization_max_error_mm']:
                write('level-localization-check.json',dict(passed=False,windows=windows))
                raise ValueError('All-size temporal localization must pass before level fitting')
            a=data[split];times=d['next_time'][:]
            if surface:lookup=surface_lookup(xy,detail,spec['declared_cup_geometry'])
            for index in np.flatnonzero(times>=times[-1]-1.-1e-7)[::2]:
                rgb=decode(d['next_obs/cup_detail_image'][int(index)],'jpeg95')
                a['images'].append(cup_crop(rgb,xy,detail))
                if surface:
                    a['surface_images'].append(surface_crop(rgb,xy,detail,spec['declared_cup_geometry']))
                    a['surface_lookup'].append(lookup)
                a['target'].append(r['label']['target']);a['xy'].append(xy)
                a['meta'].append(dict(name=spec['name'],cup_height_scale=spec['cup_height_scale'],
                    cup_diameter_scale=spec['cup_diameter_scale'],index=int(index),source_sha256=r['sha256']))
    write('level-localization-check.json',dict(passed=True,windows=windows,
        scope='Actual image localization and temporal confirmation on every declared fitting episode.'))
    for a in data.values():
        a['x']=torch.tensor(np.stack(a.pop('images')),device='cuda')
        if surface:
            a['surface_x']=torch.tensor(np.stack(a.pop('surface_images')),device='cuda')
            a['surface_lookup']=torch.tensor(np.stack(a['surface_lookup']),device='cuda',dtype=torch.float64)
        a['xy']=torch.tensor(np.asarray(a['xy']),device='cuda',dtype=torch.float32)
        a['y']=torch.tensor(a['target'],device='cuda',dtype=torch.float32)
        a['weights']=torch.tensor(balanced_frame_weights(a['meta']),device='cuda')
        a['cnn']=torch.cat([model(x) for x in a['x'].split(128)]).add(parent['cup_bias']).clamp(0,1)
        a['features']=features(a['x'],a['cnn'],a['xy'])
    return plan,parent,model,data


def metrics(prediction,a):
    error=prediction-a['y'];near=a['y']>.85
    def values(mask):
        e=error[mask];n=mask&near
        return dict(mae=float(e.abs().mean()),max_abs=float(e.abs().max()),
            near_max=float(error[n].abs().max()),near_overestimate=float(error[n].max()))
    result=values(torch.ones_like(near))
    result['sizes']={f'{h:g}/{d:g}':values(torch.tensor([
        m['cup_height_scale']==h and m['cup_diameter_scale']==d for m in a['meta']],device='cuda')) for h,d in SIZES}
    return result


def eligible(result,gate):
    return all(r['mae']<gate['level_validation_mae'] and r['near_max']<gate['level_validation_near_max_error']
               and r['near_overestimate']<gate['level_validation_near_overestimate'] for r in result['sizes'].values())


def main():
    assert not (ROOT/'static-level-fit.json').exists() and not (ROOT/'component-selection.json').exists()
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);torch.manual_seed(26110971);started=time.perf_counter()
    plan,parent,model,data=collect();gate=plan['component_gates'];tr=data['train'];va=data['validation']
    audit_path,review_path=collection_paths(ROOT,read(ROOT/'candidate.json'))
    with torch.no_grad():
        raw=metrics(va['cnn'],va);retained=metrics(predict(va['features'],parent['settled_regressor']),va)
    # Keep the existing RGB feature definition and architecture, but fit a
    # separate static candidate. Never overwrite the parent stop estimator.
    columns=list(range(7))+list(range(340,367))+list(range(604,tr['features'].shape[1]))
    xx=tr['features'][:,columns].clone();mean=xx.mean(0);scale=xx.std(0).clamp(min=.001);xx=(xx-mean)/scale
    net=torch.nn.Sequential(torch.nn.Linear(len(columns),32),torch.nn.Tanh(),
        torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1)).cuda()
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    best=None;attempt=None;history=[];steps=1200;fit_start=time.perf_counter()
    for step in range(1,steps+1):
        delta=net(xx)[:,0]
        loss=((tr['cnn']+delta-tr['y']).square()*tr['weights']).sum()+.02*(delta.square()*tr['weights']).sum()
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step%20:continue
        layers=[dict(weight=l.weight.detach().cpu().tolist(),bias=l.bias.detach().cpu().tolist()) for l in net if isinstance(l,torch.nn.Linear)]
        fitted=dict(columns=[4],mean=[0.],scale=[1.],coef=[1.],output_bias=0.,
            residual_mlp=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),layers=layers))
        with torch.no_grad():
            validation=metrics(predict(va['features'],fitted),va);training=metrics(predict(tr['features'],fitted),tr)
        passed=eligible(validation,gate) and eligible(training,gate)
        score=validation['mae']+.3*validation['near_max']
        row=dict(step=step,validation=validation,training=training,eligible=passed,score=score);history.append(row)
        if attempt is None or score<attempt['metrics']['score']:attempt=dict(fit=copy.deepcopy(fitted),metrics=row)
        if passed and (best is None or score<best['metrics']['score']):best=dict(fit=copy.deepcopy(fitted),metrics=row)
        if step%200==0:print(json.dumps(dict(step=step,eligible=passed,validation=validation)),flush=True)
    fit_seconds=time.perf_counter()-fit_start
    chosen=best or attempt
    with torch.no_grad():
        cpu_model=CupEstimator().eval();cpu_model.load_state_dict(parent['cup_model']);cpu=va['x'].cpu()
        cnn=torch.cat([cpu_model(x) for x in cpu.split(128)]).add(parent['cup_bias']).clamp(0,1)
        cpu_prediction=predict(features(cpu,cnn,va['xy'].cpu()),chosen['fit'])
        gpu_prediction=predict(va['features'],chosen['fit'])
        parity=float((cpu_prediction-gpu_prediction.cpu()).abs().max())
        independent=metrics(color_geometry_readout(va['x'],parent['surface_geometry'],va['xy']),va)
    passed=best is not None and parity<1e-4
    artifact=ROOT/'static-level-candidate.pt'
    torch.save(dict(cup_model=parent['cup_model'],cup_bias=parent['cup_bias'],settled_regressor=chosen['fit'],
        surface_geometry=parent['surface_geometry'],accepted=passed,scope='static_known_sizes_only'),artifact)
    report=dict(passed=passed,steps=steps,newly_fitted_parameters=sum(p.numel() for p in net.parameters()),
        raw_cnn_validation=raw,retained_parent_validation=retained,retained_independent_color_validation=independent,
        selected=chosen['metrics'],history=history,cpu_gpu_max_difference=parity,
        counts={s:dict(images=len(a['y']),episodes=len({m['name'] for m in a['meta']}),
            min_level=min(a['target']),max_level=max(a['target'])) for s,a in data.items()},
        fit_seconds=fit_seconds,total_seconds=time.perf_counter()-started,
        plan_sha256=sha(ROOT/'plan.json'),parent_checkpoint_sha256=sha(ROOT/'best.pt'),
        localization_checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt'),checkpoint_sha256=sha(artifact),
        audit_sha256=sha(audit_path),manual_review_sha256=sha(review_path),
        helper_sha256=sha(Path(__file__)),feature_helper_sha256=sha(Path('/workspace/src/pour_settled_model.py')),
        localization_module_sha256=sha(ROOT/'scale_rim_refiner.py'),
        device=torch.cuda.get_device_name(),autonomous_use_accepted=False,
        method='Supervised 139-feature, 32/16 tanh residual regressor over a frozen CNN. Equal size/episode weighted squared error plus 0.02 residual penalty; AdamW, 1200 steps. Whole-episode validation and per-size error gates. No RL.',
        limitations=['Static scenes, original bottle and declared cup dimensions only.',
            'Current level range does not establish >=94% or held-bottle/dynamic-stop performance.',
            'The retained independent color estimator is diagnosed, not retrained or newly accepted.',
            'No runtime activation; independent static holdout and later dynamic/clearance gates remain.'])
    write('static-level-fit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','selected','limitations','raw_cnn_validation','retained_parent_validation','retained_independent_color_validation')}),flush=True)


if __name__=='__main__':main()
