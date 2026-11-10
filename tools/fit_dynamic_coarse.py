"""Repair coarse crop selection; only the later full rim gate can accept it."""
import argparse,hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from fit_dynamic_rim import ROOT,read,sha,collect,protected_unchanged,CupLocator,centers
from scale_rim_augmentation import affine
from scale_data_admission import balanced_frame_weights

@torch.inference_mode()
def evaluate(model,a):
    rows=[]
    for start in range(0,len(a['x']),32):
        image=a['x'][start:start+32];uv,score=centers(model(image));negative=image.clone()
        for j,(px,py) in enumerate(a['u'][start:start+32].round().long().tolist()):negative[j,max(0,py-24):min(384,py+48),max(0,px-50):min(384,px+50)]=180
        _,absent=centers(model(negative));errors=(uv-a['u'][start:start+32]).norm(dim=1)*(1024/384)
        rows += [dict(m,error_native_pixels=e,score=s,synthetic_absent_score=n) for m,e,s,n in zip(a['meta'][start:start+32],errors.cpu().tolist(),score.cpu().tolist(),absent.cpu().tolist())]
    return dict(max_error_native_pixels=max(r['error_native_pixels'] for r in rows),mean_error_native_pixels=float(np.mean([r['error_native_pixels'] for r in rows])),minimum_positive_score=min(r['score'] for r in rows),maximum_synthetic_absent_score=max(r['synthetic_absent_score'] for r in rows),rows=rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('d1','d2'),default='d1');args=parser.parse_args();prefix='coarse-'+args.round
    cfg=read(ROOT/(prefix+'-protocol.json'));assert not (ROOT/(prefix+'.json')).exists() and not (ROOT/'component-selection.json').exists()
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items());protected_unchanged()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);torch.manual_seed(cfg['seed']);started=time.perf_counter()
    parent=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False);model=CupLocator().cuda()
    state=parent['bank']['coarse_state'] if not cfg.get('initialization') else torch.load(ROOT/cfg['initialization'],map_location='cpu',weights_only=False)['state']
    model.load_state_dict(state);model.eval()
    data,cal=collect(model,coarse_only=True,manifest_name=prefix+'-data.json');tr,va=data['train'],data['validation'];limit=min(cfg['maximum_native_crop_error_pixels'],parent['bank']['max_correction_pixels'])
    if cfg.get('balance_motion_groups'):
        weights=np.zeros(len(tr['meta']))
        for moving in (False,True):
            ids=[i for i,m in enumerate(tr['meta']) if bool(m['moving'])==moving]
            weights[ids]=np.asarray(balanced_frame_weights([tr['meta'][i] for i in ids]))*.5
        assert abs(weights.sum()-1)<1e-6;tr['weights']=torch.tensor(weights,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    y,x=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(96,device='cuda'),indexing='ij');history=[];best=attempt=None;steps=0;fit_started=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        if epoch in cfg['lr_steps']:
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.multinomial(tr['weights'],len(tr['x']),replacement=True).split(16):
            image=tr['x'][ids].float();target=tr['u'][ids].clone();n=len(ids)
            image,target=affine(image,target,torch.eye(2,device='cuda').repeat(n,1,1),(torch.rand(n,2,device='cuda')*2-1)*12)
            positive=torch.rand(n,device='cuda')>cfg.get('negative_probability',.25)
            for j in torch.nonzero(~positive,as_tuple=True)[0].tolist():
                px,py=target[j].round().long().tolist();image[j,max(0,py-24):min(384,py+48),max(0,px-50):min(384,px+50)]=180
            image=(image*(.95+.1*torch.rand(n,1,1,1,device='cuda'))).clamp(0,255)
            cell=(target/4).floor().long();assert (cell>=0).all() and (cell<96).all()
            gaussian=torch.exp(-((x[None]-cell[:,0,None,None])**2+(y[None]-cell[:,1,None,None])**2)/(2*1.5**2))*positive[:,None,None]
            output=model(image);logits=output[:,0];prob=logits.sigmoid();pos=gaussian==1;neg=~pos
            heat=(-(1-prob).square()*F.logsigmoid(logits)*pos-prob.square()*F.logsigmoid(-logits)*(1-gaussian)**4*neg).sum()/positive.sum().clamp(min=1)
            index=torch.arange(n,device='cuda');offset=output[index,1:,cell[:,1],cell[:,0]].sigmoid()
            offset_loss=F.smooth_l1_loss(offset[positive],(target/4-cell)[positive]) if positive.any() else output.sum()*0
            loss=heat+2*offset_loss;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        if epoch%5:continue
        model.eval();tt=evaluate(model,tr);vr=evaluate(model,va)
        margin=min(tt['minimum_positive_score'],vr['minimum_positive_score'])-max(tt['maximum_synthetic_absent_score'],vr['maximum_synthetic_absent_score'])
        ok=max(tt['max_error_native_pixels'],vr['max_error_native_pixels'])<limit and margin>cfg['absence_margin']
        row=dict(epoch=epoch,passed_for_refinement=ok,train_max_native_px=tt['max_error_native_pixels'],validation_max_native_px=vr['max_error_native_pixels'],absence_margin=margin,minimum_positive=min(tt['minimum_positive_score'],vr['minimum_positive_score']),maximum_absent=max(tt['maximum_synthetic_absent_score'],vr['maximum_synthetic_absent_score']));history.append(row);print(json.dumps(row),flush=True)
        result=dict(row=row,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},training=tt,validation=vr)
        if attempt is None or row['validation_max_native_px']<attempt['row']['validation_max_native_px']:attempt=result
        if ok and (best is None or row['validation_max_native_px']<best['row']['validation_max_native_px']):best=result
    chosen=best or attempt;model.load_state_dict(chosen['state']);model.eval();cpu=CupLocator().eval();cpu.load_state_dict(chosen['state'])
    with torch.inference_mode():
        differences=[]
        for images in va['x'].split(16):
            cpu_uv,cs=centers(cpu(images.cpu()));gpu_uv,gs=centers(model(images));differences.append(float((cpu_uv-gpu_uv.cpu()).norm(dim=1).max())*(1024/384))
        parity=max(differences)
    passed=best is not None and parity<.1
    torch.save(dict(state=chosen['state'],accepted_for_refinement=passed,accepted=False),ROOT/(prefix+'.pt'));chosen.pop('state')
    report=dict(passed_for_refinement=passed,accepted=False,selected=chosen,history=history,counts={s:dict(episodes=len({m['name'] for m in a['meta']}),frames=len(a['meta'])) for s,a in data.items()},parameters=sum(p.numel() for p in model.parameters()),steps=steps,cpu_gpu_max_native_pixels=parity,checkpoint_sha256=sha(ROOT/(prefix+'.pt')),protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),fit_seconds=time.perf_counter()-fit_started,total_seconds=time.perf_counter()-started,autonomy_accepted=False,scope='Coarse crop recovery only; 3 mm final localization, temporal, level and color gates remain required.')
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items());protected_unchanged()
    (ROOT/(prefix+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed_for_refinement=passed,selected=chosen['row'])),flush=True)
if __name__=='__main__':main()
