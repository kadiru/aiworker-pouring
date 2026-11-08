"""Fit the existing runtime stop-level readout; raw CNN errors remain reported."""
import copy,time,urllib.request
from collections import Counter
from pathlib import Path
import sys
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
import numpy as np
import torch
from fit_perception_cup import ROOT,read,sha,write,collect,localize,collect_level_data
from pour_guard_models import CupEstimator
from pour_guard_vision import color_geometry_readout,surface_features
from pour_settled_model import features,predict

def main():
    h=read_health();assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert not (ROOT/'cup-component.json').exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);torch.manual_seed(26110833);start=time.perf_counter()
    plan,seq,cal=collect();assert plan['level_prediction_kind']=='cnn_plus_supervised_stop_readout'
    settings=plan['stop_readout_fit'];gate=plan['component_gates']
    parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    source=ROOT/'rounds/task-r3-level-attempt2/cup-level-attempt.pt'
    frozen=torch.load(source,map_location='cpu',weights_only=False)
    loc,loc_report=localize(seq,cal,gate);assert loc is not None
    write('cup-localization-validation.json',loc_report)
    data=collect_level_data(seq,cal,parent);model=CupEstimator().cuda().eval();model.load_state_dict(frozen['cup_model'])
    baseline_names={e['spec']['name'] for e in seq if e['spec'].get('cup_height_scale',1.)==e['spec'].get('cup_diameter_scale',1.)==1.}
    with torch.no_grad():
        for split,a in data.items():
            a['cnn']=torch.cat([model(x) for x in a['x'].split(128)]).clamp(0,1)
            a['features']=torch.cat([features(a['x'][i:i+128],a['cnn'][i:i+128],a['xy'][i:i+128]) for i in range(0,len(a['x']),128)]).double()
            a['primary']=torch.tensor([d!='historical_task_replay' and n in baseline_names for n,d in zip(a['name'],a['domain'])],device='cuda')
            a['near']=a['primary']&(a['y']>.85)
            a['task']=torch.tensor([d=='consumed_task_detail' for d in a['domain']],device='cuda')
            a['replay']=torch.tensor([d=='historical_task_replay' for d in a['domain']],device='cuda')
    tr,va=data['train'],data['validation'];assert int((va['task']&va['near']).sum())>=15
    def metrics(values):
        error=values-va['y'];tn=va['task']&va['near']
        return dict(mae=float(error[va['primary']].abs().mean()),near_max=float(error[va['near']].abs().max()),
            near_overestimate=float(error[va['near']].max()),actual_pour_mae=float(error[va['task']].abs().mean()),
            actual_pour_near_max=float(error[tn].abs().max()),actual_pour_near_overestimate=float(error[tn].max()),
            historical_replay_mae=float(error[va['replay']].abs().mean()))
    raw=metrics(va['cnn']);count=Counter(tr['name'])
    weights=tr['y'].new_tensor([(settings['task_weight'] if d=='consumed_task_detail' else 4. if d=='native_detail' else 1.)/count[n] for n,d in zip(tr['name'],tr['domain'])])
    weights*=torch.where(tr['y']>.8,3.,1.);weights/=weights.mean()
    candidates=[];best=None;best_attempt=None
    def consider(fitted,description):
        nonlocal best,best_attempt
        with torch.no_grad():
            row=metrics(predict(va['features'],fitted));te=predict(tr['features'],fitted)-tr['y']
            row.update(training_near_max=float(te[tr['near']].abs().max()),training_near_overestimate=float(te[tr['near']].max()))
        eligible=(max(row['mae'],row['actual_pour_mae'])<gate['level_validation_mae']
            and max(row['near_max'],row['actual_pour_near_max'],row['training_near_max'])<gate['level_validation_near_max_error']
            and max(row['near_overestimate'],row['actual_pour_near_overestimate'],row['training_near_overestimate'])<gate['level_validation_near_overestimate']
            and row['historical_replay_mae']<=max(.05,raw['historical_replay_mae']+.015))
        score=row['mae']+row['actual_pour_mae']+.3*row['near_max'];record=dict(description,**row,eligible=eligible,score=score)
        candidates.append(record)
        if best_attempt is None or score<best_attempt['report']['score']:best_attempt=dict(fit=copy.deepcopy(fitted),report=record)
        if eligible and (best is None or score<best['report']['score']):best=dict(fit=copy.deepcopy(fitted),report=record)
    # The final runtime estimator already supports these RGB/position features.
    x=tr['features'];vx=va['features'];y=tr['y'].double()
    feature_sets=[list(range(60)),list(range(7))+list(range(604,x.shape[1]))]
    for columns in feature_sets:
        a=x[:,columns];mean=a.mean(0);scale=a.std(0).clamp(min=.001);mean[0]=0;scale[0]=1.;a=(a-mean)/scale
        for ridge in settings['ridge']:
            reg=torch.eye(len(columns),device='cuda',dtype=torch.float64)*ridge;reg[0,0]=1e-9
            coef=torch.linalg.solve(a.T@(a*weights[:,None])+reg,a.T@(y*weights))
            fitted=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),coef=coef.cpu().tolist(),output_bias=0.)
            consider(fitted,dict(kind='ridge',inputs=len(columns),ridge=ridge))
    # One fixed residual network on the CNN prediction, selected only by the
    # declared development validation gates; no hand-chosen output bias.
    assert settings['residual_features']=='first7_plus_exact_count_quantiles_and_spatial_shape'
    columns=list(range(7))+list(range(340,367))+list(range(604,x.shape[1]))
    xx=x[:,columns].float();mean=xx.mean(0);scale=xx.std(0).clamp(min=.001);xx=(xx-mean)/scale
    net=torch.nn.Sequential(torch.nn.Linear(len(columns),32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1)).cuda()
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    base=dict(columns=[4],mean=[0.],scale=[1.],coef=[1.],output_bias=0.)
    for step in range(1,settings['residual_steps']+1):
        delta=net(xx)[:,0];loss=((tr['cnn']+delta-tr['y']).square()*weights).mean()+.02*delta.square().mean()
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step%20:continue
        layers=[dict(weight=l.weight.detach().cpu().tolist(),bias=l.bias.detach().cpu().tolist()) for l in net if isinstance(l,torch.nn.Linear)]
        fitted=dict(base,residual_mlp=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),layers=layers))
        consider(fitted,dict(kind='cnn_residual_mlp',inputs=len(columns),widths=[32,16],step=step))
        if step%200==0:print('STOP_READOUT',candidates[-1],flush=True)
    chosen=best or best_attempt;geometry=frozen['geometry']
    with torch.no_grad():
        pred=color_geometry_readout(va['x'],geometry,va['xy']);_,_,mass=surface_features(va['x']);pred=torch.where(mass>1.,pred,0.)
        color_max=float((pred-va['y'])[va['primary']].abs().max())
        # Match CPU evaluator and actual float32 runtime decoding, not only
        # the double-precision fitting matrices.
        cpu_model=CupEstimator().eval();cpu_model.load_state_dict(frozen['cup_model'])
        cpu_x=va['x'].cpu();cpu_xy=va['xy'].cpu()
        cpu_cnn=torch.cat([cpu_model(b) for b in cpu_x.split(128)]).clamp(0,1)
        cpu_features=features(cpu_x,cpu_cnn,cpu_xy)
        cpu_prediction=predict(cpu_features,chosen['fit'])
        gpu_prediction=predict(va['features'].float(),chosen['fit'])
        parity=float((cpu_prediction-gpu_prediction.cpu()).abs().max())
        precision=float((predict(va['features'],chosen['fit'])-gpu_prediction).abs().max())
        feature_difference=(cpu_features.double()-va['features'].cpu()).abs().max(0).values
        numerical=dict(cpu_gpu_max_difference=parity,float32_float64_max_difference=precision,
            cpu_gpu_raw_cnn_max_difference=float((cpu_cnn-va['cnn'].cpu()).abs().max()),
            feature_max_differences=feature_difference.tolist(),
            cpu_validation=metrics(cpu_prediction.cuda()),gpu_float32_validation=metrics(gpu_prediction))
    numerical_passed=parity<1e-4 and precision<1e-4
    passed=best is not None and color_max<gate['geometry_validation_max_error'] and numerical_passed
    report=dict(passed=passed,scope='baseline_cup_only',prediction_kind=plan['level_prediction_kind'],selected=chosen['report'],raw_cnn_validation=raw,
        independent_color_max_error=color_max,numerical_check_passed=numerical_passed,numerical=numerical,source_cnn_sha256=sha(source),candidates=candidates,
        counts={s:dict(images=len(a['y']),episodes=len(set(a['name'])),actual_task_images=int(a['task'].sum()),near_task_images=int((a['task']&a['near']).sum())) for s,a in data.items()},
        newly_fitted_parameters=len(chosen['fit']['coef'])+sum(len(l['bias'])+sum(map(len,l['weight'])) for l in chosen['fit'].get('residual_mlp',{}).get('layers',[])),
        method='Supervised episode-balanced ridge candidates and one fixed 32/16 tanh residual network over frozen CNN/RGB surface features and image-localized XY; weighted squared error plus residual penalty. Gates apply to the final runtime stop-level estimator, with raw CNN errors retained. No RL, future-final targets, truth-centered crops or task-score/threshold changes.',
        helper_sha256=sha(Path(__file__)),dataset_helper_sha256=sha(ROOT/'fit_perception_cup.py'),label_helper_sha256=sha(ROOT/'perception_task_labels.py'))
    write('cup-level-validation.json',report)
    level=dict(cup_model=frozen['cup_model'],cup_bias=0.,surface_geometry=geometry,settled_regressor=chosen['fit'])
    torch.save(dict(**level,accepted=passed,selected=chosen['report']),ROOT/'cup-level-attempt.pt')
    component=dict(passed=passed,scope='baseline_cup_only',localization_passed=True,level_passed=passed,source_sha256=sha(Path(__file__)),total_seconds=time.perf_counter()-start,device=torch.cuda.get_device_name())
    if passed:
        torch.save(dict(localization=loc,level=level,detail_calibration=cal),ROOT/'cup-component.pt');component['checkpoint_sha256']=sha(ROOT/'cup-component.pt')
    write('cup-component.json',component);print(component,flush=True)
def read_health():
    import json
    return json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
if __name__=='__main__':main()
