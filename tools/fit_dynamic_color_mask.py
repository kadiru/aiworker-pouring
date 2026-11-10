"""Supervised independent color-mask fitting; no prior reserved images."""
import hashlib,json,sys,time
from pathlib import Path
import torch
sys.path[:0]=['/workspace/src','/workspace']
import fit_scale_level as data_module
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1');data_module.ROOT=ROOT
from fit_scale_level import collect,metrics,eligible,read,sha,write
from scale_level_cnn import ScaleLevelCNN,level_context,predict_size_cnn
from scale_color_mask import mask_image,predict_mask
from scale_rim_augmentation import affine

@torch.inference_mode()
def predictions(model,a):return torch.cat([model(a['mask'][i:i+128],a['context'][i:i+128]) for i in range(0,len(a['mask']),128)])

def main():
    assert not (ROOT/'color-mask-r2.json').exists();protocol=read(ROOT/'color-mask-protocol-r2.json')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);torch.manual_seed(protocol['seed']);started=time.perf_counter()
    plan,_,_,data=collect();specs={s['name']:s for s in plan['collect']};gate=plan['component_gates'];gates=read(ROOT/'protocol.json')['gates']
    primary=torch.load(ROOT/'primary-level.pt',map_location='cpu',weights_only=False)
    for a in data.values():
        a['geometry']=[specs[m['name']]['declared_cup_geometry'] for m in a['meta']]
        a['mask'],valid=mask_image(a['x']);assert valid.all();a['context']=level_context(a['mask'],a['geometry'],a['xy'])
        a['primary']=predict_size_cnn(a['x'],a['xy'],a['geometry'],primary)
    tr=data['train'];va=data['validation'];model=ScaleLevelCNN().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=1e-4)
    best=attempt=None;history=[];steps=0;fit_start=time.perf_counter()
    for epoch in range(1,protocol['epochs']+1):
        model.train()
        if epoch in (151,241):
            for group in opt.param_groups:group['lr']/=3
        for ids in torch.multinomial(tr['weights'],len(tr['x']),replacement=True).split(64):
            x=tr['mask'][ids].float();n=len(ids);matrix=torch.eye(2,device='cuda').repeat(n,1,1)
            shift=(torch.rand(n,2,device='cuda')*2-1)*protocol['crop_jitter_pixels']
            x,_=affine(x,torch.zeros(n,2,device='cuda'),matrix,shift)
            error=model(x,tr['context'][ids])-tr['y'][ids];loss=error.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        if epoch%5:continue
        model.eval();tp=predictions(model,tr);vp=predictions(model,va);training=metrics(tp,tr);validation=metrics(vp,va)
        disagreement=max(float((tp-tr['primary']).abs().max()),float((vp-va['primary']).abs().max()))
        ok=eligible(training,gate) and eligible(validation,gate) and disagreement<gates['crosscheck_max_disagreement']
        score=max(v['mae'] for v in validation['sizes'].values())+.3*validation['near_max']
        row=dict(epoch=epoch,training=training,validation=validation,max_primary_disagreement=disagreement,passed=ok,score=score);history.append(row)
        chosen=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},report=row)
        if attempt is None or score<attempt['report']['score']:attempt=chosen
        if ok and (best is None or score<best['report']['score']):best=chosen
        if epoch%30==0:print(json.dumps(dict(epoch=epoch,passed=ok,worst_size_mae=max(v['mae'] for v in validation['sizes'].values()),disagreement=disagreement)),flush=True)
    fit_seconds=time.perf_counter()-fit_start;chosen=best or attempt
    identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(chosen['state'].items()))).hexdigest()
    artifact=dict(state=chosen['state'],identity=identity,accepted=False,scope='development_only_color_mask')
    gpu,valid=predict_mask(va['x'],va['xy'],va['geometry'],artifact);cpu,cv=predict_mask(va['x'].cpu(),va['xy'].cpu(),va['geometry'],artifact)
    parity=float((gpu.cpu()-cpu).abs().max());cpu_metrics=metrics(cpu.cuda(),va)
    absent,visible=predict_mask(torch.full_like(va['x'],180),va['xy'],va['geometry'],artifact)
    assert valid.all() and cv.all() and not visible.any() and torch.isnan(absent).all()
    passed=best is not None and parity<1e-4 and eligible(cpu_metrics,gate);artifact['accepted']=passed;torch.save(artifact,ROOT/'color-mask-r2.pt')
    report=dict(passed=passed,selected=chosen['report'],history=history,cpu_validation=cpu_metrics,cpu_gpu_difference=parity,
        absent_color_rejected=True,epochs=protocol['epochs'],optimizer_steps=steps,parameters=sum(p.numel() for p in model.parameters()),
        training_episodes=126,validation_episodes=18,fit_seconds=fit_seconds,total_seconds=time.perf_counter()-started,
        checkpoint_sha256=sha(ROOT/'color-mask-r2.pt'),primary_checkpoint_sha256=sha(ROOT/'primary-level.pt'),
        protocol_sha256=sha(ROOT/'color-mask-protocol-r2.json'),module_sha256=sha(ROOT/'scale_color_mask.py'),helper_sha256=sha(Path(__file__)),
        scope='Independent random initialization and binary red-mask input only; geometry/pose shared. No primary prediction or RGB weights feed this network. Development-only selection, no autonomy accepted.')
    write('color-mask-r2.json',report);print(json.dumps(dict(passed=passed,epoch=chosen['report']['epoch'],parity=parity)),flush=True)
if __name__=='__main__':main()
