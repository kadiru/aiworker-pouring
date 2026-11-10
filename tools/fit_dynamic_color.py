"""Train an independent color cross-check on the prior development data only."""
import argparse,copy,json,sys,time
from pathlib import Path
import torch
sys.path[:0]=['/workspace/src','/workspace']
import fit_scale_level as data_module
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1');data_module.ROOT=ROOT
from fit_scale_level import collect,metrics,eligible,read,sha,write
from scale_color_check import feature_matrix,predict_color
from scale_surface_geometry import envelope_features
from scale_level_readout import predict_readout
from scale_level_cnn import predict_size_cnn

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',default='r0',choices=('r0','r1'));args=parser.parse_args()
    report_name=f'color-{args.round}.json';checkpoint_name=f'color-{args.round}.pt'
    assert not (ROOT/report_name).exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);start=time.perf_counter();protocol_path=ROOT/('protocol.json' if args.round=='r0' else f'color-protocol-{args.round}.json');protocol=read(protocol_path)
    plan,parent,_,data=collect(surface=True);cfg=protocol['color_fit'];gate=plan['component_gates']
    specs={s['name']:s for s in plan['collect']};tr=data['train'];va=data['validation']
    for a in data.values():a['geometry']=[specs[m['name']]['declared_cup_geometry'] for m in a['meta']]
    names=list(dict.fromkeys(m['name'] for m in tr['meta']));assert len(names)==126
    ids=[torch.tensor([i for i,m in enumerate(tr['meta']) if m['name']==n],device='cuda') for n in names]
    target=torch.stack([tr['y'][i].double().mean() for i in ids]);bias=target.mean()
    primary=torch.load(ROOT/'primary-level.pt',map_location='cpu',weights_only=False)
    with torch.inference_mode():cnn=predict_size_cnn(va['x'],va['xy'],va['geometry'],primary)
    candidates=[];best=None;attempt=None;fit_start=time.perf_counter()
    def consider(fit,description,x,v,cpu):
        nonlocal best,attempt
        tp=predict_readout(x,fit);vp=predict_readout(v,fit);cp=predict_readout(cpu,fit)
        train=metrics(tp,tr);validation=metrics(vp,va);cpu_validation=metrics(cp.cuda(),va)
        parity=float((vp.cpu()-cp).abs().max());disagreement=float((vp-cnn).abs().max())
        ok=all(eligible(r,gate) for r in (train,validation,cpu_validation)) and parity<1e-4 and disagreement<protocol['gates']['crosscheck_max_disagreement']
        score=max(r['mae'] for r in validation['sizes'].values())+.3*validation['near_max']
        row=dict(description,training=train,validation=validation,cpu_validation=cpu_validation,cpu_gpu_difference=parity,
            max_primary_disagreement=disagreement,passed=ok,score=score)
        candidates.append(row);candidate=dict(fit=copy.deepcopy(fit),report=row)
        if attempt is None or score<attempt['report']['score']:attempt=candidate
        if ok and (best is None or score<best['report']['score']):best=candidate
    for kind in cfg['feature_sets']:
        x,xt=feature_matrix(tr['x'],tr['xy'],tr['geometry'],kind,tr['surface_x'],tr['surface_lookup'])
        v,xv=feature_matrix(va['x'],va['xy'],va['geometry'],kind,va['surface_x'],va['surface_lookup'])
        cpu,xc=feature_matrix(va['x'].cpu(),va['xy'].cpu(),va['geometry'],kind,va['surface_x'].cpu(),va['surface_lookup'].cpu())
        assert xt.all() and xv.all() and xc.all()
        means=torch.stack([x[i].mean(0) for i in ids]);mean=means.mean(0);scale=means.std(0).clamp(min=.001);z=(means-mean)/scale
        common=dict(features=kind,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),bias=float(bias))
        for penalty in cfg['ridge_penalties']:
            coef=torch.linalg.solve(z.T@z+torch.eye(z.shape[1],device='cuda',dtype=torch.float64)*penalty,z.T@(target-bias))
            consider(dict(common,kind='ridge',coef=coef.cpu().tolist()),dict(kind='ridge',features=kind,penalty=penalty,parameters=z.shape[1]+1),x,v,cpu)
        distance=(z[:,None]-z[None]).square().mean(-1)
        for length in cfg['kernel_length_scales']:
            kernel=torch.exp(-distance/(2*length**2))
            for penalty in cfg['kernel_ridge_penalties']:
                alpha=torch.linalg.solve(kernel+torch.eye(len(z),device='cuda',dtype=torch.float64)*penalty,target-bias)
                consider(dict(common,kind='rbf',centers=z.cpu().tolist(),length_scale=length,alpha=alpha.cpu().tolist()),
                    dict(kind='rbf',features=kind,penalty=penalty,length_scale=length,parameters=len(z)+1),x,v,cpu)
    chosen=best or attempt;artifact=dict(color_readout=chosen['fit'],accepted=best is not None,scope='existing_development_only')
    # Shared inference and an explicit no-color rejection use the selected path.
    actual,valid=predict_color(va['x'],va['xy'],va['geometry'],artifact,va['surface_x'],va['surface_lookup'])
    absent,absence_valid=predict_color(torch.full_like(va['x'],180),va['xy'],va['geometry'],artifact,torch.full_like(va['surface_x'],180),va['surface_lookup'])
    assert valid.all() and not absence_valid.any() and torch.isnan(absent).all()
    path=metrics(actual,va);assert abs(path['mae']-chosen['report']['validation']['mae'])<1e-10
    torch.save(artifact,ROOT/checkpoint_name)
    report=dict(passed=best is not None,selected=chosen['report'],candidates=candidates,fit_seconds=time.perf_counter()-fit_start,total_seconds=time.perf_counter()-start,
        training_episodes=len(names),validation_episodes=18,absent_color_rejected=True,protocol_sha256=sha(protocol_path),
        primary_checkpoint_sha256=sha(ROOT/'primary-level.pt'),locator_checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt'),
        checkpoint_sha256=sha(ROOT/checkpoint_name),source_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'scale_color_check.py'),
        method='CNN-independent color-envelope and declared-geometry ridge/RBF regression; CUDA float64, equal episode means, MSE regularization. No reserved images, no RL.',
        scope='Development baseline only; high-fill, motion, independent tests and runtime integration still gated.')
    write(report_name,report);print(json.dumps(dict(passed=report['passed'],selected=chosen['report'])),flush=True)
if __name__=='__main__':main()
