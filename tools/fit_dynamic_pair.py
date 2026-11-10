"""Fit RGB and independent color-mask level models on reviewed development data.

Both regress measured height with squared error. The mask network never consumes
RGB-model predictions or weights. Agreement is a selection gate, not a loss.
"""
import argparse,hashlib,json,time
import numpy as np
import torch
from dynamic_level_data import ROOT,collect,sha,read
from fit_scale_level import metrics,eligible
from scale_level_cnn import ScaleLevelCNN,level_context,predict_size_cnn
from scale_color_mask import mask_image,predict_mask
from scale_rim_augmentation import affine

def subset(a,ids):
    return dict(y=a['y'][ids],meta=[m for m,keep in zip(a['meta'],ids.tolist()) if keep])

@torch.inference_mode()
def scores(pred,a):
    result={'all':metrics(pred,a)}
    for moving in (False,True):
        ids=torch.tensor([bool(m.get('moving'))==moving for m in a['meta']],device='cuda')
        if ids.any():result['moving' if moving else 'static']=metrics(pred[ids],subset(a,ids))
    return result

def passes(result,gates):return all(eligible(m,gates) for m in result.values())

@torch.inference_mode()
def infer(model,a):return torch.cat([model(a['input'][i:i+128],a['context'][i:i+128]) for i in range(0,len(a['input']),128)])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r3','r4'),default='r3');args=parser.parse_args()
    prefix='pair-'+args.round;config=read(ROOT/(prefix+'-protocol.json'))
    assert not (ROOT/(prefix+'.json')).exists() and not (ROOT/'component-selection.json').exists()
    assert config['protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(sha(ROOT/n)==v for n,v in config['frozen_files'].items())
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    torch.manual_seed(config['seed']);started=time.perf_counter();protocol,parent,data=collect(config['parts'],config.get('locator','cup-neural-localization.pt'),config.get('locator_selection','scope-selection.json'));gates=protocol['gates']
    tr=data['train'];va=data['validation'];assert any(m.get('moving') for m in va['meta'])
    if config.get('balance_motion_groups'):
        from scale_data_admission import balanced_frame_weights
        weights=np.zeros(len(tr['meta']))
        for moving in (False,True):
            ids=[i for i,m in enumerate(tr['meta']) if bool(m.get('moving'))==moving]
            weights[ids]=np.asarray(balanced_frame_weights([tr['meta'][i] for i in ids]))*.5
        assert abs(weights.sum()-1)<1e-6;tr['weights']=torch.tensor(weights,device='cuda')
    initial=torch.load(ROOT/config['rgb_initialization'],map_location='cpu',weights_only=False)
    summaries={};models={};primary_predictions={}
    for kind in ('rgb','color'):
        model=ScaleLevelCNN().cuda()
        if kind=='rgb':model.load_state_dict(initial['size_cnn_state'])
        elif config.get('color_initialization'):
            path=ROOT/config['color_initialization'];assert sha(path)==config['color_initialization_sha256']
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
        for a in data.values():
            a['input']=a['x']
            if kind=='color':
                a['input'],valid=mask_image(a['x']);assert valid.all(),'Required cup color absent in a development frame'
            a['context']=level_context(a['input'],a['geometry'],a['xy'])
        opt=torch.optim.AdamW(model.parameters(),lr=config[kind]['lr'],weight_decay=1e-4)
        best=attempt=None;history=[];steps=0;fit_started=time.perf_counter()
        for epoch in range(1,config[kind]['epochs']+1):
            model.train()
            if epoch in config[kind]['lr_steps']:
                for group in opt.param_groups:group['lr']/=3
            for ids in torch.multinomial(tr['weights'],len(tr['x']),replacement=True).split(64):
                x=tr['input'][ids].float();n=len(ids)
                shift=(torch.rand(n,2,device='cuda')*2-1)*config['jitter_pixels']
                x,_=affine(x,x.new_zeros(n,2),torch.eye(2,device='cuda').repeat(n,1,1),shift)
                error=model(x,tr['context'][ids])-tr['y'][ids];loss=error.square().mean()
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
            if epoch%5:continue
            model.eval();tp=infer(model,tr);vp=infer(model,va);training=scores(tp,tr);validation=scores(vp,va)
            disagreement=None if kind=='rgb' else max(float((tp-primary_predictions['train']).abs().max()),float((vp-primary_predictions['validation']).abs().max()))
            ok=passes(training,gates) and passes(validation,gates) and (disagreement is None or disagreement<gates['crosscheck_max_disagreement'])
            score=max(m['mae'] for group in validation.values() for m in group['sizes'].values())+.3*max(m['near_max'] for m in validation.values())
            row=dict(epoch=epoch,training=training,validation=validation,max_primary_disagreement=disagreement,passed=ok,score=score);history.append(row)
            chosen=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},row=row)
            if attempt is None or score<attempt['row']['score']:attempt=chosen
            if ok and (best is None or score<best['row']['score']):best=chosen
            if epoch%40==0:print(json.dumps(dict(kind=kind,epoch=epoch,passed=ok,score=score,disagreement=disagreement)),flush=True)
        chosen=best or attempt;model.load_state_dict(chosen['state']);model.eval()
        identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(chosen['state'].items()))).hexdigest()
        artifact=dict(accepted=False,scope='development_only_known_sizes_no_autonomy')
        if kind=='rgb':artifact.update(size_cnn_state=chosen['state'],size_cnn_identity=identity)
        else:artifact.update(state=chosen['state'],identity=identity)
        with torch.inference_mode():
            if kind=='rgb':
                gpu=predict_size_cnn(va['x'],va['xy'],va['geometry'],artifact);cpu=predict_size_cnn(va['x'].cpu(),va['xy'].cpu(),va['geometry'],artifact)
                primary_predictions={s:infer(model,a) for s,a in data.items()};absent_ok=None
            else:
                gpu,valid=predict_mask(va['x'],va['xy'],va['geometry'],artifact);cpu,cv=predict_mask(va['x'].cpu(),va['xy'].cpu(),va['geometry'],artifact)
                absent,visible=predict_mask(torch.full_like(va['x'],180),va['xy'],va['geometry'],artifact)
                absent_ok=bool(valid.all() and cv.all() and not visible.any() and torch.isnan(absent).all())
            parity=float((cpu-gpu.cpu()).abs().max());path_difference=float((gpu-infer(model,va)).abs().max());cpu_scores=scores(cpu.cuda(),va)
        accepted=best is not None and parity<1e-4 and path_difference<1e-4 and passes(cpu_scores,gates) and absent_ok is not False
        artifact['accepted']=accepted;file=ROOT/(prefix+'-'+kind+'.pt');torch.save(artifact,file);models[kind]=sha(file)
        rows=[]
        for split,a in data.items():
            pred=infer(model,a).cpu().tolist()
            rows += [dict(m,split=split,target=y,prediction=v,error=v-y) for m,y,v in zip(a['meta'],a['target'],pred)]
        report=dict(passed=accepted,selected=chosen['row'],history=history,rows=rows,cpu_scores=cpu_scores,cpu_gpu_difference=parity,inference_path_difference=path_difference,
            absent_color_rejected=absent_ok,parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,fit_seconds=time.perf_counter()-fit_started,
            checkpoint_sha256=sha(file),helper_sha256=sha(ROOT/'fit_dynamic_pair.py'),data_helper_sha256=sha(ROOT/'dynamic_level_data.py'))
        (ROOT/(prefix+'-'+kind+'.json')).write_text(json.dumps(report,indent=2)+'\n');summaries[kind]={k:report[k] for k in ('passed','parameters','optimizer_steps','fit_seconds','cpu_gpu_difference','inference_path_difference','checkpoint_sha256')}
    result=dict(passed=all(r['passed'] for r in summaries.values()),models=summaries,
        counts={s:dict(episodes=len({m['name'] for m in a['meta']}),frames=len(a['y'])) for s,a in data.items()},
        protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),scene_protocol_sha256=sha(ROOT/'protocol.json'),total_seconds=time.perf_counter()-started,
        method='Separate supervised RGB and binary-color-mask regressors; equal-size/equal-episode sampled squared error with AdamW. No RL or primary-to-color distillation.',autonomy_accepted=False)
    assert all(sha(ROOT/n)==v for n,v in config['frozen_files'].items())
    (ROOT/(prefix+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
