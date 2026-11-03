"""Small recurrent visuomotor BC baseline, fitted inside the owned simulator.

Actor inputs are two RGB views and measured joint position/velocity only.
Simulator quantities supply labels and evaluation, never actor observations.
"""
import io
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/workspace/learning/A-20261102-v1')
SIZE = 128
CHUNK = 10
SEED = 20261102
PHASES = ['Approaching bottle','Closing fingers','Lifting and checking grip',
          'Pouring from gripper','Returning upright','Setting bottle down','Releasing bottle']
SCALE = np.array([.08]*7+[.5]*4, dtype=np.float32)


def pixels(frame, codec='raw'):
    if codec == 'jpeg95':
        frame = np.asarray(Image.open(io.BytesIO(frame.tobytes())).convert('RGB'))
    return np.asarray(Image.fromarray(frame).resize((SIZE,SIZE), Image.Resampling.BILINEAR)).copy()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,32,3,2,1))
        y,x=torch.meshgrid(torch.linspace(-1,1,8),torch.linspace(-1,1,8),indexing='ij')
        self.register_buffer('coords',torch.stack([x.flatten(),y.flatten()],-1))

    def forward(self, x):
        z=self.conv(x)
        attention=z.flatten(2).softmax(-1)
        return torch.cat([(attention @ self.coords).flatten(1),z.mean((2,3))],-1)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.task=Encoder()
        self.wrist=Encoder()
        self.fuse=nn.Sequential(nn.Linear(96*2+22,192),nn.ReLU())
        self.rnn=nn.GRU(192,192,batch_first=True)
        self.action=nn.Linear(214,7*CHUNK)
        self.grip=nn.Linear(214,CHUNK)
        self.stop=nn.Linear(192,3*CHUNK)
        self.done=nn.Linear(192,1)
        self.phase=nn.Linear(192,len(PHASES))
        self.fill=nn.Linear(192,2)
        self.register_buffer('q_mean',torch.zeros(11))
        self.register_buffer('q_std',torch.ones(11))

    def forward(self, task,wrist,q,v,h=None):
        b,t=q.shape[:2]
        def encode(enc,images):
            x=images.flatten(0,1).permute(0,3,1,2).float()/255.
            return enc(x).reshape(b,t,-1)
        x=torch.cat([encode(self.task,task),encode(self.wrist,wrist),
                     (q-self.q_mean)/self.q_std,v.clamp(-5,5)/2.],-1)
        x,h=self.rnn(self.fuse(x),h)
        self.features=torch.cat([x,(q-self.q_mean)/self.q_std,v.clamp(-5,5)/2.],-1)
        self.predicted_grip=self.grip(self.features)
        self.predicted_fill=self.fill(x)
        return (self.action(self.features).reshape(b,t,CHUNK,7),
                self.stop(x).reshape(b,t,CHUNK,3),self.done(x).squeeze(-1),self.phase(x),h)


def decode_actions(model,arm):
    targets=model.q_mean[:7]+model.q_std[:7]*arm
    closure=model.predicted_grip.sigmoid()
    closure=torch.where(closure<.05,0.,torch.where(closure>.95,1.,closure))
    return torch.cat([targets,closure[...,None].expand(*closure.shape,4)*(np.pi/4)],-1)


class TemporalEnsemble:
    """Average overlapping neural forecasts in the current predicted stop mode."""
    def __init__(self):
        self.plans=[];self.step=0;self.members=0

    def apply(self,chunk,stopped_forecast,stopped=False,reset=False):
        if reset: self.plans=[]
        self.plans.append((self.step,chunk.copy(),stopped_forecast.copy()))
        self.plans=[p for p in self.plans if self.step-p[0]<CHUNK]
        values=[];weights=[]
        for created,targets,modes in self.plans:
            age=self.step-created
            if bool(modes[age])==bool(stopped):
                values.append(targets[age]);weights.append(np.exp(-age/3.))
        self.members=len(values)
        self.step+=1
        return np.average(values,axis=0,weights=weights) if values else chunk[0].copy()


def load_episodes():
    import sys
    sys.path.append('/workspace/pydeps51')
    import h5py
    from validate_pour_recording import validate
    split={'train':[],'validation':[]}
    manifest=[]
    # Only frozen names qualify. Calibration, replay and test cannot enter fitting.
    frozen=json.loads((ROOT/'frozen-plan.json').read_text())
    allowed={s['name']:part for part in split for s in frozen[part]}
    paths=sorted(Path('/workspace/outputs/fountain').glob('*/A*/episode.hdf5'))
    # Runtime folders begin with a numeric result index.
    paths=sorted(Path('/workspace/outputs/fountain').glob('*/*A_*/episode.hdf5'))
    latest={}
    for path in paths:
        with h5py.File(path,'r') as f:
            spec=json.loads(f.attrs['spec'])
            if spec['name'] in allowed:
                latest[spec['name']]=path
    for name,path in latest.items():
        report=validate(path)
        measured=report['metrics']['bottle_result']
        spill=max(measured['spilled_ml']+measured['airborne_ml'],measured.get('ever_table_ml',0.))
        prefix=report.get('task_scope')=='approach_lift_prefix'
        limit=0. if allowed[name]=='train' else .05
        eligible=report.get('corrective_training_eligible',False) if prefix else report['training_eligible'] and spill<=limit
        if not eligible:
            manifest.append(dict(name=name,path=str(path),accepted=False,
                                 physical_success=report['physical_success'],spill_ml=spill))
            continue
        with h5py.File(path,'r') as f:
            d=f['data/demo_0'];codec=f.attrs.get('image_encoding','raw')
            episode={camera:torch.from_numpy(np.stack([pixels(frame,codec)
                     for frame in d['obs/'+camera+'_image']])) for camera in ('task','wrist')}
            for dest,key in [('q','obs/joint_pos'),('v','obs/joint_vel'),('a','actions')]:
                if dest=='a' and 'teacher_actions' in d: key='teacher_actions'
                episode[dest]=torch.from_numpy(d[key][:].astype(np.float32))
            reason=[s.decode() if isinstance(s,bytes) else str(s) for s in d['ground_truth/stop_reason'][:]]
            episode['stop']=torch.tensor([{'cup_near_full':1,'bottle_empty':2}.get(s,0) for s in reason])
            phases=[s.decode() if isinstance(s,bytes) else str(s) for s in d['ground_truth/phase'][:]]
            episode['phase']=torch.tensor([PHASES.index(s) if s in PHASES else 0 for s in phases])
            spec=json.loads(f.attrs['spec'])
            cup=np.r_[spec.get('cup_initial_ml',0.)/72.8,d['ground_truth/cup_inventory_fraction'][:-1]]
            source=np.r_[spec['initial_ml'],d['ground_truth/bottle_ml'][:-1]]/100.
            episode['fill']=torch.tensor(np.stack([cup,source],-1),dtype=torch.float32)
            episode['done']=torch.zeros(len(reason));episode['done'][-5:]=1.
            if prefix: episode['done'].zero_()
            episode['corrective']=prefix
            episode['n']=len(reason)
            indices=torch.arange(episode['n'])[:,None]+torch.arange(CHUNK)[None,:]
            episode['future_valid']=indices<episode['n']
            indices=indices.clamp(max=episode['n']-1)
            episode['future_a']=episode['a'][indices]
            episode['future_stop']=episode['stop'][indices]
            episode['path']=str(path)
            split[allowed[name]].append(episode)
            manifest.append(dict(name=name,path=str(path),accepted=True,split=allowed[name],transitions=len(reason),
                                 action_label_key='teacher_actions' if 'teacher_actions' in d else 'actions',
                                 task_scope='approach_lift_prefix' if prefix else 'full_pour'))
    if len(split['train'])<6 or len(split['validation'])<2:
        raise RuntimeError('Need at least six reviewed successful train and two validation episodes')
    (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return split


def batch(episodes, rng, count=6, length=48, augment=False):
    selected=[]
    corrective=[e for e in episodes if e.get('corrective')]
    whole=[e for e in episodes if not e.get('corrective')]
    for _ in range(count):
        pool=corrective if corrective and rng.random()<.5 else (whole or episodes)
        e=pool[int(rng.integers(len(pool)))]
        start=0 if rng.random()<(.6 if e.get('corrective') else .2) else int(rng.integers(max(1,e['n']-length+1)))
        item={k:e[k][start:start+length] for k in ('task','wrist','q','v','a','stop','done','phase','fill','future_a','future_stop','future_valid')}
        item['mask']=torch.ones(length,dtype=torch.bool)
        if start: item['mask'][:8]=False
        selected.append(item)
    result={k:torch.stack([s[k] for s in selected]).cuda(non_blocking=False) for k in selected[0]}
    if augment:
        for c in ('task','wrist'):
            x=result[c].float()
            gain=torch.empty((count,1,1,1,1),device='cuda').uniform_(.97,1.03)
            result[c]=(x*gain+torch.randn_like(x)*1.5).clamp(0,255)
        result['q']=result['q']+torch.randn_like(result['q'])*.003
        result['v']=result['v']+torch.randn_like(result['v'])*.01
    return result


def loss(model,b):
    a,s,d,p,_=model(b['task'],b['wrist'],b['q'],b['v'])
    target=(b['future_a'][...,:7]-model.q_mean[:7])/model.q_std[:7]
    # Random interior windows get eight history steps. True episode starts are
    # supervised from their first action, matching zero-hidden-state inference.
    mask=b['mask']
    future_mask=mask[...,None]&b['future_valid']
    arm=F.mse_loss(a[future_mask],target[future_mask])
    physical=(model.q_mean[:7]+model.q_std[:7]*a-b['future_a'][...,:7])/torch.as_tensor(SCALE[:7],device=a.device)
    arm=arm+.05*physical[future_mask].square().mean()
    closure=(b['future_a'][...,7:].mean(-1)/(np.pi/4)).clamp(0,1)
    grip=F.binary_cross_entropy_with_logits(model.predicted_grip[future_mask],closure[future_mask])
    action=arm+.05*grip
    stop=F.cross_entropy(s[future_mask],b['future_stop'][future_mask])
    done=F.binary_cross_entropy_with_logits(d[mask],b['done'][mask],pos_weight=torch.tensor(8.,device=a.device))
    phase=F.cross_entropy(p[mask],b['phase'][mask])
    fill=F.mse_loss(model.predicted_fill[mask],b['fill'][mask])
    total=action+.05*stop+.015*done+.015*phase+.1*fill
    return total,dict(action=float(action.detach()),grip=float(grip.detach()),stop=float(stop.detach()),done=float(done.detach()),fill=float(fill.detach()))


@torch.inference_mode()
def offline_validation(model,episodes):
    """Teacher-observation diagnostics and RGB ablation, not task success."""
    result={}
    for mode in ('normal_rgb','blank_rgb'):
        action_error=[];fill_error=[];stop_correct=[];events=[]
        for e in episodes:
            h=None;first_stop=None;correct_reason=None;votes=0;last_label=0;latched=False
            ensemble=TemporalEnsemble()
            for i in range(0,e['n'],32):
                part={k:e[k][i:i+32][None].cuda() for k in ('task','wrist','q','v')}
                if mode=='blank_rgb':
                    part['task'].zero_();part['wrist'].zero_()
                a,s,_,_,h=model(part['task'],part['wrist'],part['q'],part['v'],h)
                chunks=decode_actions(model,a)[0].cpu().numpy()
                all_probs=s[0].softmax(-1).cpu().numpy()
                probs=all_probs[:,0]
                predictions=[]
                for j,prob in enumerate(probs):
                    label=int(np.argmax(prob));just_stopped=False
                    votes=(votes+1 if label==last_label else 1) if label and prob[label]>.8 else 0
                    last_label=label
                    if not latched and votes>=2:
                        latched=True;just_stopped=True;first_stop=(i+j)*.05;correct_reason=label
                    predictions.append(ensemble.apply(chunks[j],all_probs[j].argmax(-1)!=0,
                                                      stopped=latched,reset=just_stopped))
                action_error.extend(np.asarray(predictions)-e['a'][i:i+32].numpy())
                fill_error.extend(abs(model.predicted_fill[0].cpu()-e['fill'][i:i+32]).numpy())
                stop_correct.extend((probs.argmax(-1)==e['stop'][i:i+32].numpy()).tolist())
            truth=torch.nonzero(e['stop']>0).flatten()
            events.append(dict(path=e['path'],first_predicted_stop_s=first_stop,
               predicted_reason=correct_reason,teacher_first_stop_s=float(truth[0])*.05 if len(truth) else None))
        result[mode]=dict(action_rmse_deg_per_joint=np.degrees(np.sqrt(np.mean(np.square(action_error),axis=0))).tolist(),
            fill_mae_fraction=np.mean(fill_error,axis=0).tolist(),stop_class_accuracy=float(np.mean(stop_correct)),events=events)
    (ROOT/'offline-validation.json').write_text(json.dumps(result,indent=2)+'\n')


@torch.inference_mode()
def initialize_absolute_head(model,episodes):
    """Training-only ridge initialization of absolute targets from features."""
    xs=[];ys=[];valid=[];model.eval()
    for e in episodes:
        h=None
        for i in range(0,e['n'],32):
            part={k:e[k][i:i+32][None].cuda() for k in ('task','wrist','q','v')}
            _,_,_,_,h=model(part['task'],part['wrist'],part['q'],part['v'],h)
            f=model.features[0]
            xs.append(torch.cat([f,torch.ones((len(f),1),device='cuda')],-1))
            a=e['future_a'][i:i+32].cuda()
            arm=(a[...,:7]-model.q_mean[:7])/model.q_std[:7]
            closure=(a[...,7:].mean(-1)/(np.pi/4)).clamp(.01,.99)
            ys.append(torch.cat([arm,torch.logit(closure)[...,None]],-1))
            valid.append(e['future_valid'][i:i+32].cuda())
    x=torch.cat(xs).double();y=torch.cat(ys).double()
    valid=torch.cat(valid)
    ridge=1e-3*torch.eye(x.shape[-1],device='cuda',dtype=torch.float64)
    weights=[];errors=[]
    for k in range(CHUNK):
        xx=x[valid[:,k]];yy=y[valid[:,k],k]
        w=torch.linalg.solve(xx.T@xx+ridge,xx.T@yy)
        weights.append(w);errors.append(float((xx@w-yy).square().mean()))
    model.action.weight.copy_(torch.stack([w[:-1,:7].T for w in weights]).reshape(CHUNK*7,-1))
    model.action.bias.copy_(torch.stack([w[-1,:7] for w in weights]).flatten())
    model.grip.weight.copy_(torch.stack([w[:-1,7] for w in weights]))
    model.grip.bias.copy_(torch.stack([w[-1,7] for w in weights]))
    return float(np.mean(errors))


def train(state, updates=4000):
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED);np.random.seed(SEED)
    assert torch.cuda.device_count()==1, 'Training must use the isolated owned GPU'
    assert '4070' in torch.cuda.get_device_name(0)
    device=dict(name=torch.cuda.get_device_name(0),visible_cuda_devices=torch.cuda.device_count(),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    (ROOT/'training-device.json').write_text(json.dumps(device,indent=2)+'\n')
    rng=np.random.default_rng(SEED)
    state.update(phase='Loading reviewed demonstration images',learning=dict(stage='loading'))
    data=load_episodes()
    model=Policy().cuda()
    q=torch.cat([e['q'] for e in data['train']])
    model.q_mean.copy_(q.mean(0));model.q_std.copy_(q.std(0).clamp(min=.05))
    warm=next((ROOT/name for name in ('best-bc-v4.pt','best-bc-v3.pt','best-bc-v2.pt','best-bc-v1.pt') if (ROOT/name).exists()),ROOT/'absent.pt')
    if warm.exists():
        old=torch.load(warm,map_location='cuda',weights_only=False)['model']
        shared={k:v for k,v in old.items() if not k.startswith(('action.','grip.','stop.'))}
        model.load_state_dict(shared,strict=False)
        with torch.no_grad():
            model.stop.weight.copy_(old['stop.weight'].repeat(CHUNK,1))
            model.stop.bias.copy_(old['stop.bias'].repeat(CHUNK))
    initialization=initialize_absolute_head(model,data['train'])
    (ROOT/'absolute-head-initialization.json').write_text(json.dumps(dict(training_only=True,ridge_mse=initialization),indent=2)+'\n')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    best=float('inf');history=[];started=time.monotonic()
    for step in range(1,updates+1):
        model.train();b=batch(data['train'],rng,augment=True)
        optimizer.zero_grad(set_to_none=True)
        total,parts=loss(model,b)
        if not torch.isfinite(total):
            raise FloatingPointError(f'Non-finite training loss at update {step}')
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1 or step%100==0:
            model.eval()
            with torch.no_grad():
                val_rng=np.random.default_rng(SEED+9999)
                vals=[loss(model,batch(data['validation'],val_rng))[1] for _ in range(8)]
            val=float(np.mean([v['action'] for v in vals]))
            row=dict(step=step,train=parts,validation_action=val,elapsed_s=time.monotonic()-started)
            history.append(row)
            state.update(phase=f'Training visual policy {step}/{updates}',progress=step/updates,
                         learning=dict(stage='training',eta_s=(time.monotonic()-started)/step*(updates-step),**row))
            (ROOT/'training-history.json').write_text(json.dumps(history,indent=2)+'\n')
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                seed=SEED,size=SIZE,arm_loss_scale_rad=SCALE[:7].tolist(),validation_action=val,
                model_version='chunked_bc_v5',action_space='absolute action chunks with phase-consistent temporal ensemble',
                action_chunk=CHUNK,ensemble_decay_steps=3.,
                training_augmentation=dict(rgb_gain=[.97,1.03],rgb_noise_std=1.5,joint_position_std_rad=.003,joint_velocity_std_rad_s=.01),
                training_device=device,
                policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dataset_manifest_sha256=hashlib.sha256((ROOT/'dataset-manifest.json').read_bytes()).hexdigest(),
                scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
                inputs=['task RGB','wrist RGB','joint position','joint velocity'],
                train_episodes=len(data['train']),validation_episodes=len(data['validation']),
                full_training_episodes=sum(not e.get('corrective') for e in data['train']),
                corrective_prefixes=sum(e.get('corrective',False) for e in data['train']),
                selection='Lowest validation action loss; autonomous rollouts reported separately')
            torch.save(payload,ROOT/'last.partial.pt');(ROOT/'last.partial.pt').replace(ROOT/'last.pt')
            if val<best:
                best=val
                torch.save(payload,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    state.update(phase='Visual policy fit complete',learning=dict(stage='fit_complete',
                 updates=updates,best_validation_action=best,elapsed_s=time.monotonic()-started))
    model.load_state_dict(torch.load(ROOT/'best.pt',weights_only=False)['model'])
    model.eval();offline_validation(model,data['validation'])
    del optimizer,model,data
    torch.cuda.empty_cache()
    return str(ROOT/'best.pt')


class Actor:
    def __init__(self,checkpoint):
        torch.set_num_threads(4)
        self.saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.model=Policy().cuda().eval();self.model.load_state_dict(self.saved['model'])
        self.hidden=None;self.stop_votes=0;self.done_votes=0;self.last_stop_label=0
        self.steps=0;self.clamps=0;self.stop_latched=None
        self.arm_limits=None
        self.ensemble=TemporalEnsemble()
        self.diagnostics=[]

    @torch.inference_mode()
    def act(self,observation,robot,t):
        # Explicit allowlist. No scenario specification, volume, pose or clock is
        # passed to the neural network. t is used only to timestamp its decision.
        images=[torch.from_numpy(pixels(observation[c+'_encoded'],'jpeg95'))[None,None].cuda() for c in ('task','wrist')]
        q=torch.from_numpy(observation['joint_pos'])[None,None].cuda()
        v=torch.from_numpy(observation['joint_vel'])[None,None].cuda()
        a,s,d,p,self.hidden=self.model(*images,q,v,self.hidden)
        probs=s[0,0,0].softmax(-1).cpu().numpy()
        label=int(np.argmax(probs));just_stopped=False
        if label and probs[label]>.8:
            self.stop_votes=self.stop_votes+1 if label==self.last_stop_label else 1
        else:
            self.stop_votes=0
        self.last_stop_label=label
        if self.stop_latched is None and self.stop_votes>=2:
            self.stop_latched=['','cup_near_full','bottle_empty'][label]
            robot.reason=self.stop_latched;robot.stop_time=t;just_stopped=True
        chunks=decode_actions(self.model,a)[0,0].cpu().numpy()
        desired=self.ensemble.apply(chunks,s[0,0].argmax(-1).cpu().numpy()!=0,
            stopped=self.stop_latched is not None,reset=just_stopped)
        delta=desired-observation['joint_pos']
        bounded=np.clip(delta,-np.r_[[.12]*7,[.85]*4],np.r_[[.12]*7,[.85]*4])
        targets=observation['joint_pos']+bounded
        if self.arm_limits is None:
            self.arm_limits=np.radians([[state.GetPrim().GetAttribute('physics:'+key+'Limit').Get()
                                         for key in ('lower','upper')] for state in robot.states[:7]]).T
        limited=np.clip(targets[:7],self.arm_limits[0],self.arm_limits[1])
        clamped=not np.allclose(delta,bounded) or not np.allclose(targets[:7],limited)
        self.clamps+=int(clamped)
        targets[:7]=limited
        targets[7:]=np.clip(targets[7:],0,np.pi/4)
        done=float(d[0,0].sigmoid())
        released=(np.max(abs(observation['joint_pos'][7:]))<.10 and
                  np.max(abs(observation['joint_vel']))<.08)
        self.done_votes=self.done_votes+1 if self.stop_latched and released else 0
        robot.phase='Visual BC: '+PHASES[int(p[0,0].argmax())]
        self.steps+=1
        self.diagnostics.append(dict(t=t,stop_prob=probs.tolist(),done_probability=done,
            predicted_fill=self.model.predicted_fill[0,0].cpu().tolist(),
            delta_clamped=bool(clamped),ensemble_members=self.ensemble.members))
        return np.degrees(targets)

    def complete(self):
        # Proprioceptive end-of-episode test only. This never changes actions or
        # supplies a pour-stop signal. Physical set-down is scored separately.
        return self.done_votes>=10
