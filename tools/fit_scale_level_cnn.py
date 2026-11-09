"""Fine-tune RGB level features on all declared sizes with crop jitter."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from fit_scale_level import ROOT,read,sha,write,collect,metrics,eligible
from scale_data_admission import collection_paths
from scale_level_cnn import ScaleLevelCNN,level_context,predict_size_cnn
from scale_rim_augmentation import affine
from pour_guard_vision import color_geometry_readout

@torch.inference_mode()
def predictions(model,a):return torch.cat([model(a['x'][i:i+128],a['context'][i:i+128]) for i in range(0,len(a['x']),128)])

def main():
    assert not (ROOT/'static-level-fit.json').exists() and not (ROOT/'component-selection.json').exists()
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r9','r10'),default='r9');args=parser.parse_args()
    protocol_path=ROOT/f'level-refit-{args.round}-plan.json'
    protocol=read(protocol_path);m=read(ROOT/'candidate.json');ap,rp=collection_paths(ROOT,m)
    assert protocol['plan_sha256']==sha(ROOT/'plan.json') and protocol['audit_sha256']==sha(ap) and protocol['manual_review_sha256']==sha(rp)
    assert all(sha(ROOT/n)==v for n,v in protocol['frozen_locator'].items())
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);torch.manual_seed(protocol['seed']);started=time.perf_counter()
    plan,parent,_,data=collect();specs={s['name']:s for s in plan['collect']};tr=data['train'];va=data['validation'];gate=plan['component_gates']
    for a in data.values():
        a['geometry']=[specs[r['name']]['declared_cup_geometry'] for r in a['meta']]
        a['context']=level_context(a['x'],a['geometry'],a['xy'])
    model=ScaleLevelCNN().cuda();model.cup.load_state_dict(parent['cup_model'])
    if protocol.get('fine_initialization'):
        source=ROOT/protocol['fine_initialization'];assert sha(source)==protocol['fine_initialization_sha256']
        model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['size_cnn_state'])
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    history=[];best=None;attempt=None;steps=0;fitstart=time.perf_counter()
    for epoch in range(1,protocol['epochs']+1):
        model.train()
        if epoch in (121,181):
            for group in opt.param_groups:group['lr']/=3
        for ids in torch.multinomial(tr['weights'],len(tr['x']),replacement=True).split(protocol['batch_size']):
            x=tr['x'][ids].float();n=len(ids);rng=np.random.default_rng(protocol['seed']+steps);cfg=protocol['augmentation']
            angle=np.deg2rad(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees'],n));scale=rng.uniform(cfg['scale_min'],cfg['scale_max'],n)
            matrix=np.stack((np.cos(angle)*scale,-np.sin(angle)*scale,np.sin(angle)*scale,np.cos(angle)*scale),1).reshape(n,2,2)
            shift=rng.uniform(-cfg['translation_pixels'],cfg['translation_pixels'],(n,2))
            x,_=affine(x,x.new_zeros((n,2)),torch.tensor(matrix,device='cuda',dtype=torch.float32),torch.tensor(shift,device='cuda',dtype=torch.float32))
            x=(x*(cfg['brightness_min']+(cfg['brightness_max']-cfg['brightness_min'])*torch.rand(n,1,1,1,device='cuda'))).clamp(0,255)
            error=model(x,tr['context'][ids])-tr['y'][ids];loss=error.square().mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        if epoch%5:continue
        model.eval();vp=predictions(model,va);tp=predictions(model,tr);validation=metrics(vp,va);training=metrics(tp,tr)
        ok=eligible(validation,gate) and eligible(training,gate);score=max(r['mae'] for r in validation['sizes'].values())+.3*validation['near_max']
        row=dict(epoch=epoch,validation=validation,training=training,accuracy_passed=ok,score=score);history.append(row)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};candidate=dict(state=state,report=row)
        if attempt is None or score<attempt['report']['score']:attempt=candidate
        if ok and (best is None or score<best['report']['score']):best=candidate
        if epoch%20==0:print(json.dumps(dict(epoch=epoch,eligible=ok,validation_worst_size_mae=max(r['mae'] for r in validation['sizes'].values()),near_max=validation['near_max'],training_worst_size_mae=max(r['mae'] for r in training['sizes'].values()))),flush=True)
    fit_seconds=time.perf_counter()-fitstart;chosen=best or attempt;model.load_state_dict(chosen['state']);model.eval()
    identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(chosen['state'].items()))).hexdigest()
    artifact=dict(cup_model=parent['cup_model'],cup_bias=parent['cup_bias'],size_cnn_state=chosen['state'],size_cnn_identity=identity,
        surface_geometry=parent['surface_geometry'],scope='static_known_sizes_only')
    with torch.inference_mode():
        gpu=predict_size_cnn(va['x'],va['xy'],va['geometry'],artifact)
        cpu=predict_size_cnn(va['x'].cpu(),va['xy'].cpu(),va['geometry'],artifact)
        parity=float((gpu.cpu()-cpu).abs().max());path_difference=float((gpu-predictions(model,va)).abs().max())
        cpu_validation=metrics(cpu.cuda(),va);passed=best is not None and parity<1e-4 and path_difference<1e-4 and eligible(cpu_validation,gate)
        independent=metrics(color_geometry_readout(va['x'],parent['surface_geometry'],va['xy']),va)
    artifact['accepted']=passed;torch.save(artifact,ROOT/'static-level-candidate.pt');rows=[]
    for split,a in data.items():
        pred=predictions(model,a).cpu().tolist()
        for meta,target,value in zip(a['meta'],a['target'],pred):rows.append(dict(meta,split=split,target=target,prediction=value,error=value-target))
    report=dict(passed=passed,selected=chosen['report'],history=history,rows=rows,cpu_validation=cpu_validation,
        epochs=protocol['epochs'],optimizer_steps=steps,newly_fitted_parameters=sum(p.numel() for p in model.parameters()),
        cpu_gpu_max_difference=parity,inference_path_max_difference=path_difference,retained_independent_color_validation=independent,
        counts={s:dict(images=len(a['y']),episodes=len({r['name'] for r in a['meta']}),min_level=min(a['target']),max_level=max(a['target'])) for s,a in data.items()},
        fit_seconds=fit_seconds,total_seconds=time.perf_counter()-started,device=torch.cuda.get_device_name(),
        plan_sha256=sha(ROOT/'plan.json'),refit_protocol_sha256=sha(protocol_path),parent_checkpoint_sha256=sha(ROOT/'best.pt'),
        localization_checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt'),checkpoint_sha256=sha(ROOT/'static-level-candidate.pt'),
        audit_sha256=sha(ap),manual_review_sha256=sha(rp),helper_sha256=sha(Path(__file__)),data_helper_sha256=sha(ROOT/'fit_scale_level.py'),
        localization_module_sha256=sha(ROOT/'scale_rim_refiner.py'),level_module_sha256=sha(ROOT/'scale_level_readout.py'),
        surface_module_sha256=sha(ROOT/'scale_surface_geometry.py'),cnn_module_sha256=sha(ROOT/'scale_level_cnn.py'),
        method=protocol['training'],autonomous_use_accepted=False)
    write('static-level-fit.json',report);print(json.dumps(dict(passed=passed,epoch=chosen['report']['epoch'],cpu_gpu_max_difference=parity,validation=chosen['report']['validation'])),flush=True)
if __name__=='__main__':main()
