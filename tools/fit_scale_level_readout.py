"""Size-aware supervised level refit; frozen rim/CNN, untouched reserved scenes."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from fit_scale_level import ROOT,read,sha,write,collect,metrics,eligible
from scale_data_admission import collection_paths
from scale_level_readout import level_features,predict_readout,predict_static_level,predict_surface_readout
from scale_surface_geometry import envelope_features
from pour_guard_models import CupEstimator
from pour_guard_vision import color_geometry_readout
from pour_settled_model import predict

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r7','r8'),default='r7');args=parser.parse_args()
    protocol_path=ROOT/f'level-refit-{args.round}-plan.json';surface=args.round=='r8'
    assert not (ROOT/'static-level-fit.json').exists() and not (ROOT/'component-selection.json').exists()
    protocol=read(protocol_path);m=read(ROOT/'candidate.json');ap,rp=collection_paths(ROOT,m)
    assert protocol['plan_sha256']==sha(ROOT/'plan.json') and protocol['audit_sha256']==sha(ap) and protocol['manual_review_sha256']==sha(rp)
    assert all(sha(ROOT/n)==v for n,v in protocol['frozen_locator'].items())
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);start=time.perf_counter()
    plan,parent,model,data=collect(surface=surface);gate=plan['component_gates'];specs={s['name']:s for s in plan['collect']}
    tr=data['train'];va=data['validation'];cpu_model=CupEstimator().eval();cpu_model.load_state_dict(parent['cup_model'])
    with torch.inference_mode():
        cpu_image=va['x'].cpu();cpu_xy=va['xy'].cpu()
        cpu_cnn=torch.cat([cpu_model(x) for x in cpu_image.split(128)]).add(parent['cup_bias']).clamp(0,1)
        for a in data.values():a['geometry']=[specs[m['name']]['declared_cup_geometry'] for m in a['meta']]
        raw=metrics(va['cnn'],va);retained=metrics(predict(va['features'],parent['settled_regressor']),va)
    names=list(dict.fromkeys(m['name'] for m in tr['meta']));assert len(names)==90
    index=[torch.tensor([i for i,m in enumerate(tr['meta']) if m['name']==n],device='cuda') for n in names]
    targets=torch.stack([tr['y'][ids].double().mean() for ids in index]);bias=targets.mean()
    candidates=[];best=None;attempt=None;fitstart=time.perf_counter();cached={}
    def consider(fitted,description,x,v,cpu):
        nonlocal best,attempt
        with torch.inference_mode():
            tp=predict_readout(x,fitted);vp=predict_readout(v,fitted);cp=predict_readout(cpu,fitted)
            training=metrics(tp,tr);validation=metrics(vp,va);cpu_validation=metrics(cp.cuda(),va)
            parity=float((cp-vp.cpu()).abs().max())
        accuracy=eligible(training,gate) and eligible(validation,gate) and eligible(cpu_validation,gate)
        ok=accuracy and parity<1e-4
        score=max(r['mae'] for r in validation['sizes'].values())+.3*validation['near_max']
        row=dict(description,training=training,validation=validation,cpu_validation=cpu_validation,accuracy_passed=accuracy,cpu_gpu_max_difference=parity,eligible=ok,score=score)
        candidates.append(row)
        chosen=dict(fit=copy.deepcopy(fitted),report=row)
        if attempt is None or (score,validation['mae'])<(attempt['report']['score'],attempt['report']['validation']['mae']):attempt=chosen
        if ok and (best is None or (score,validation['mae'])<(best['report']['score'],best['report']['validation']['mae'])):best=chosen
        print(json.dumps({k:v for k,v in row.items() if k not in ('training','validation','cpu_validation')}|dict(validation_worst_size_mae=max(r['mae'] for r in validation['sizes'].values()),near_max=validation['near_max'])),flush=True)
    for kind in protocol['feature_sets']:
        with torch.inference_mode():
            if kind=='physical_surface':
                x,baseline_tr,valid_tr=envelope_features(tr['surface_x'],tr['surface_lookup'],tr['geometry'],tr['xy'])
                v,baseline_va,valid_va=envelope_features(va['surface_x'],va['surface_lookup'],va['geometry'],va['xy'])
                cpu,_,valid_cpu=envelope_features(va['surface_x'].cpu(),va['surface_lookup'].cpu(),va['geometry'],cpu_xy)
                assert valid_tr.all() and valid_va.all() and valid_cpu.all(), 'Surface evidence incomplete; no fit or test acceptance'
            else:
                x=level_features(tr['x'],tr['cnn'],tr['xy'],tr['geometry'],kind)
                v=level_features(va['x'],va['cnn'],va['xy'],va['geometry'],kind)
                cpu=level_features(cpu_image,cpu_cnn,cpu_xy,va['geometry'],kind)
        means=torch.stack([x[ids].mean(0) for ids in index]);mean=means.mean(0);scale=means.std(0).clamp(min=.001)
        z=(means-mean)/scale;response=targets-bias;cached[kind]=(x,v,cpu)
        common=dict(features=kind,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),bias=float(bias))
        for penalty in protocol['ridge_penalties']:
            coefficient=torch.linalg.solve(z.T@z+torch.eye(z.shape[1],device='cuda',dtype=torch.float64)*penalty,z.T@response)
            fitted=dict(common,kind='ridge',coef=coefficient.cpu().tolist())
            consider(fitted,dict(kind='ridge',features=kind,dimensions=z.shape[1],penalty=penalty,learned_parameters=z.shape[1]+1),x,v,cpu)
        distance=(z[:,None]-z[None]).square().mean(-1)
        for length in protocol['kernel_length_scales']:
            kernel=torch.exp(-distance/(2*length**2))
            for penalty in protocol['kernel_ridge_penalties']:
                alpha=torch.linalg.solve(kernel+torch.eye(len(z),device='cuda',dtype=torch.float64)*penalty,response)
                fitted=dict(common,kind='rbf',centers=z.cpu().tolist(),length_scale=length,alpha=alpha.cpu().tolist())
                consider(fitted,dict(kind='rbf',features=kind,dimensions=z.shape[1],length_scale=length,penalty=penalty,learned_parameters=len(alpha)+1),x,v,cpu)
    fit_seconds=time.perf_counter()-fitstart;chosen=best or attempt;passed=best is not None
    artifact=dict(cup_model=parent['cup_model'],cup_bias=parent['cup_bias'],size_level_readout=chosen['fit'],
        surface_geometry=parent['surface_geometry'],accepted=passed,scope='static_known_sizes_only')
    with torch.inference_mode():
        actual=predict_surface_readout(va['surface_x'],va['surface_lookup'],va['geometry'],va['xy'],artifact) if surface else predict_static_level(va['x'],va['cnn'],va['xy'],va['geometry'],artifact)
        cached_prediction=predict_readout(cached[chosen['fit']['features']][1],chosen['fit'])
        path_difference=float((actual-cached_prediction).abs().max());assert path_difference<1e-10
        independent=metrics(color_geometry_readout(va['x'],parent['surface_geometry'],va['xy']),va)
    torch.save(artifact,ROOT/'static-level-candidate.pt')
    rows=[]
    for split,a in data.items():
        with torch.inference_mode():
            prediction=(predict_surface_readout(a['surface_x'],a['surface_lookup'],a['geometry'],a['xy'],artifact) if surface else predict_static_level(a['x'],a['cnn'],a['xy'],a['geometry'],artifact)).cpu().tolist()
        for meta,target,value in zip(a['meta'],a['target'],prediction):rows.append(dict(meta,split=split,target=target,prediction=value,error=value-target))
    report=dict(passed=passed,selected=chosen['report'],candidates=candidates,rows=rows,candidate_count=len(candidates),eligible_candidates=sum(r['eligible'] for r in candidates),
        newly_fitted_parameters=chosen['report']['learned_parameters'],fitting_episodes=names,closed_form_solves=len(candidates),optimizer_steps=0,
        raw_cnn_validation=raw,retained_parent_validation=retained,retained_independent_color_validation=independent,
        cpu_gpu_max_difference=chosen['report']['cpu_gpu_max_difference'],inference_path_max_difference=path_difference,
        counts={s:dict(images=len(a['y']),episodes=len({m['name'] for m in a['meta']}),min_level=min(a['target']),max_level=max(a['target'])) for s,a in data.items()},
        fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),
        plan_sha256=sha(ROOT/'plan.json'),refit_protocol_sha256=sha(protocol_path),parent_checkpoint_sha256=sha(ROOT/'best.pt'),
        localization_checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt'),checkpoint_sha256=sha(ROOT/'static-level-candidate.pt'),
        audit_sha256=sha(ap),manual_review_sha256=sha(rp),helper_sha256=sha(Path(__file__)),data_helper_sha256=sha(ROOT/'fit_scale_level.py'),
        localization_module_sha256=sha(ROOT/'scale_rim_refiner.py'),level_module_sha256=sha(ROOT/'scale_level_readout.py'),
        surface_module_sha256=sha(ROOT/'scale_surface_geometry.py'),
        method=('Supervised geometric color-surface regression. High-resolution color envelope quantiles are converted to approximate physical heights through projected cup rings; regularized readout learns remaining rendering/refraction bias. ' if surface else 'Supervised size-aware RGB feature regression. Declared dimensions and image-localized pose augment color quantiles, shape and frozen CNN features. ')+'Episode-mean CUDA float64 ridge/RBF candidates; unchanged whole-episode per-size gates and CPU/GPU parity. No RL.',autonomous_use_accepted=False)
    if surface:
        report.update(unfitted_surface_validation=metrics(baseline_va,va),unfitted_surface_training=metrics(baseline_tr,tr),surface_evidence_valid=True)
    write('static-level-fit.json',report)
    print(json.dumps(dict(passed=passed,selected={k:v for k,v in chosen['report'].items() if k not in ('training','validation','cpu_validation')},eligible_candidates=report['eligible_candidates'],fit_seconds=fit_seconds)),flush=True)
if __name__=='__main__':main()
