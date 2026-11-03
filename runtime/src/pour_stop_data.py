"""Image-only liquid labels, executed expert pouring targets and learned primitives."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.append('/workspace/pydeps51')
import h5py
from pour_learning import pixels
from validate_pour_recording import decode
from pour_stop_study import ROOT


def crops(task,wrist):
    cup=np.asarray(Image.fromarray(task[172:300,198:310]).resize((96,96))).copy()
    source=task.copy();source[180:304,214:310]=180
    source=np.asarray(Image.fromarray(source[88:354,60:236]).resize((128,128))).copy()
    wrist=np.asarray(Image.fromarray(wrist).resize((128,128))).copy()
    return cup,source,wrist


def _raw_perception_data():
    arrays={s:{k:[] for k in ('cup','source','wrist','fill','inventory','cup_valid','left','empty','present')} for s in ('train','validation')}
    manifest=[]
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*/episode.hdf5')):
        with h5py.File(path,'r') as f:
            spec=json.loads(f.attrs['spec']);split=spec.get('study_split')
            if (split not in ('train','validation','test') or spec.get('name','').startswith('S_A_') or
                spec.get('spacing')!=.0018 or spec.get('dt')!=.0025 or
                f.attrs.get('camera_rig')!='A_task_wrist_v4' or f.attrs.get('image_encoding')!='jpeg95'):continue
            # Prior tests are now used evidence: perception labels only, never bad actions.
            dest='validation' if split=='validation' else 'train'
            d=f['data/demo_0'];n=len(d['actions']);indices=np.arange(0,n,3)
            samples=json.loads((path.parent/'samples.json').read_text());assert len(samples)==n
            pose=d['ground_truth/bottle_pose'][:];xyz=pose[:,:3];x,y,z,w=pose[:,3:].T
            up=np.stack([2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)],axis=-1)
            center=xyz+.05*up;T=d['next_obs/task_world_T_camera'][:];K=f['calibration/task/intrinsics'][:]
            local=np.einsum('ni,nij->nj',center-T[:,:3,3],T[:,:3,:3]);depth=-local[:,2]
            u=K[0,0]*local[:,0]/np.maximum(depth,.001)+K[0,2];v=K[1,2]-K[1,1]*local[:,1]/np.maximum(depth,.001)
            present=(depth>.01)&(u>60)&(u<236)&(v>88)&(v<354)&(xyz[:,2]>=1.035)
            left=d['ground_truth/bottle_ml'][:];inventory=d['ground_truth/cup_inventory_fraction'][:]
            fill=np.array([r['cup_fill_fraction'] for r in samples])
            unobservable=0
            for i in indices:
                c,s,wr=crops(decode(d['next_obs/task_image'][i],'jpeg95'),decode(d['next_obs/wrist_image'][i],'jpeg95'))
                rgb=c.astype(float);red=(rgb[:,:,0]>1.4*rgb[:,:,1]+6)&(rgb[:,:,0]>1.2*rgb[:,:,2]+5)
                cup_valid=bool(fill[i]<.15 or red.sum()>=8)
                unobservable+=int(not cup_valid)
                for key,value in zip(('cup','source','wrist','fill','inventory','cup_valid','left','empty','present'),
                    (c,s,wr,np.clip(fill[i],0,1),inventory[i],float(cup_valid),np.clip(left[i]/100.,0,1),float(left[i]<=max(.25,.01*spec['initial_ml'])),float(present[i]))):
                    arrays[dest][key].append(value)
            manifest.append(dict(path=str(path),name=spec['name'],original_split=split,perception_split=dest,
                frames=len(indices),unobservable_cup_frames=unobservable,fullness_label='measured surface height / rim height',used_prior_test_for_perception_only=split=='test'))
    data={s:{k:torch.from_numpy(np.asarray(v,dtype=np.uint8 if k in ('cup','source','wrist') else np.float32))
             for k,v in a.items()} for s,a in arrays.items()}
    (ROOT/'perception-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(data['train']['fill'])>1000 and len(data['validation']['fill'])>100
    return data


def perception_data():
    cache=ROOT/'perception-cache.pt'
    data=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else _raw_perception_data()
    if not cache.exists():torch.save(data,cache)
    manifest=json.loads((ROOT/'perception-manifest.json').read_text())
    # Failed global meshes can contradict particle-height labels. Fit cup level
    # only on the admitted expert/prefix recordings; retain failure images for
    # source-presence/empty negatives. These masks never enter inference.
    accepted={e['name'] for e in json.loads(Path('/workspace/learning/A-recovery-20261102-v1/dataset-manifest.json').read_text()) if e.get('accepted')}
    clean={s:[] for s in data};gates={s:[] for s in data}
    for e in manifest:
        rows=json.loads((Path(e['path']).parent/'samples.json').read_text())[::3];s=e['perception_split']
        assert len(rows)==e['frames']
        clean[s].extend([e['name'] in accepted]*len(rows))
        gates[s].extend(r['bottle_tilt_deg']>=95 and r['source_outflow_ml_s']<3 and r['phase']=='Pouring from gripper' for r in rows)
    quality={}
    for s in data:
        assert len(clean[s])==len(data[s]['fill'])
        observable=data[s]['cup_valid'].clone();clean_mask=torch.tensor(clean[s])
        # Low-height failed states are useful splash hard negatives. High
        # particle-height labels from broken global meshes stay excluded.
        reliable=clean_mask|(data[s]['fill']<.65)
        data[s]['cup_valid']*=reliable
        data[s]['empty_gate']=torch.tensor(gates[s],dtype=torch.float32)
        data[s]['source_valid']=torch.ones_like(data[s]['fill'])
        quality[s]=dict(total=len(clean[s]),admitted_reference_frames=int(clean_mask.sum()),
            cup_supervised_frames=int(data[s]['cup_valid'].sum()),unobservable_raw_frames=int((observable<.5).sum()),
            excluded_unreliable_cup_labels=int((~reliable).sum()),
            source_calibration_gate='GT tilt>=95 degrees, outflow<3 nominal mL/s, active expert pour; offline proxy for runtime proprio/visual gating')
    (ROOT/'perception-label-quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    # Explicit perception correction from the already-consumed first native
    # validation. Never consume S_A_test or actions from this failed rollout.
    extra=[]
    for p in Path('/workspace/outputs/fountain').glob('*/*S_A_validation_00*/episode.hdf5'):
        with h5py.File(p,'r') as f:
            if f.attrs['status']!='complete':continue
            rows=json.loads((p.parent/'samples.json').read_text());d=f['data/demo_0'];n=len(d['actions'])
            for i in range(0,n,2):
                if rows[i]['cup_fill_fraction']>=.65:continue
                c,s,w=crops(decode(d['next_obs/task_image'][i],'jpeg95'),decode(d['next_obs/wrist_image'][i],'jpeg95'))
                row=dict(cup=c,source=s,wrist=w,fill=rows[i]['cup_fill_fraction'],inventory=rows[i]['cup_inventory_fraction'],
                    cup_valid=1.,left=rows[i]['bottle_ml']/100,empty=0.,present=0.,empty_gate=0.,source_valid=0.)
                extra.append(row)
    if extra:
        for key in data['train']:
            add=torch.from_numpy(np.asarray([e[key] for e in extra],dtype=np.uint8 if key in ('cup','source','wrist') else np.float32))
            data['train'][key]=torch.cat([data['train'][key],add])
        # Those source labels are deliberately masked; only cup hard negatives.
    (ROOT/'corrective-perception.json').write_text(json.dumps(dict(frames=len(extra),source='consumed S_A_validation_00 only',
        purpose='low-cup splash hard negatives; source labels masked; actions unused; six tests excluded'),indent=2)+'\n')
    return data


def motion_data():
    manifest=json.loads(Path('/workspace/learning/A-recovery-20261102-v1/dataset-manifest.json').read_text())
    result={'train':[],'validation':[]};primitive=[]
    for entry in manifest:
        if not entry.get('accepted') or entry.get('corrective_prefix'):continue
        path=Path(entry['path'])
        with h5py.File(path,'r') as f:
            d=f['data/demo_0'];n=len(d['actions']);t=d['time'][:];q=d['obs/joint_pos'][:];a=d['actions'][:]
            phases=np.array([x.decode() for x in d['ground_truth/phase'][:]])
            stopped=d['ground_truth/stop_command'][:].astype(bool);prior=np.r_[False,stopped[:-1]]
            supervised=d['expert_supervision'][:].astype(bool) if 'expert_supervision' in d else np.ones(n,bool)
            take=(phases=='Pouring from gripper')&~prior&supervised
            idx=np.flatnonzero(take)
            if not len(idx):continue
            qlift=q[np.argmin(abs(t-3.25)),:7]
            e=dict(path=str(path),name=entry['name'],qlift=qlift, q=q[idx],v=d['obs/joint_vel'][:][idx],
                   delta=a[idx,:7]-q[idx,:7],task=[],wrist=[],cup=[],source=[],source_wrist=[])
            for i in idx:
                task=decode(d['obs/task_image'][i],'jpeg95');wrist=decode(d['obs/wrist_image'][i],'jpeg95')
                c,s,wr=crops(task,wrist)
                for k,value in zip(('task','wrist','cup','source','source_wrist'),(pixels(task),pixels(wrist),c,s,wr)):e[k].append(value)
            for k in ('task','wrist','cup','source','source_wrist'):e[k]=np.stack(e[k])
            result[entry['split']].append(e)
            if entry['split']!='train':continue
            stops=np.flatnonzero(stopped)
            if not len(stops):continue
            first=int(stops[0]);relative=t-t[first];grip=a[:,7:].mean(-1)/(np.pi/4)
            release=np.flatnonzero((relative>0)&(grip<.99));opened=np.flatnonzero((relative>0)&(grip<.01))
            if not len(release) or not len(opened):continue
            r0=int(release[0]);r1=int(opened[0]);ret=np.flatnonzero((phases=='Returning upright')&(relative>0))
            place=np.flatnonzero((phases=='Setting bottle down')&(relative>0))
            if not len(ret) or not len(place):continue
            return_end=float(relative[place[0]]);place_duration=float(relative[r0]-return_end)
            qp=a[first:place[0]+1,:7];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(qp,axis=0),axis=1))]
            warp=np.interp(np.linspace(0,return_end,101),relative[first:place[0]+1],arc/max(arc[-1],1e-6))
            profile=np.interp(np.linspace(t[r0],t[r1],101),t,grip)
            cup=d['ground_truth/cup_ml'][:]
            flow=max(0.,float((cup[first]-cup[max(0,first-6)])/max(t[first]-t[max(0,first-6)],.05)))
            runoff=float(np.clip((cup[-1]-cup[first])/max(flow,.1),0,1)) if cup[-1]>60 and flow>1 else None
            move=np.flatnonzero((np.arange(n)>=r1)&(np.linalg.norm(a[:,:7]-a[r0,:7],axis=1)>np.radians(.2)))
            home=np.flatnonzero((np.arange(n)>=r1)&(np.linalg.norm(a[:,:7]-a[0,:7],axis=1)<np.radians(.2)))
            if not len(move) or not len(home):continue
            retreat_s=float(t[home[0]]-t[max(r1,int(move[0])-1)])
            if retreat_s<=0:continue
            starts={p:float(t[np.flatnonzero(phases==p)[0]]-t[0]) for p in ('Closing fingers','Lifting and checking grip','Pouring from gripper') if (phases==p).any()}
            primitive.append(dict(path=str(path),return_s=return_end,place_s=place_duration,
                release_s=float(t[r1]-t[r0]),retreat_s=retreat_s,runoff_s=runoff,warp=warp.tolist(),grip=profile.tolist(),starts=starts))
    assert len(primitive)>=10
    pure=[p for p in primitive if 'Lifting and checking grip' in p['starts']]
    fitted=dict(method='time-normalized least-squares mean of executed expert profiles; reverse own measured arm history',
        return_s=float(np.median([p['return_s'] for p in primitive])),place_s=float(np.median([p['place_s'] for p in primitive])),
        release_s=float(np.median([p['release_s'] for p in primitive])),retreat_s=float(np.median([p['retreat_s'] for p in primitive])),
        runoff_s=float(np.median([p['runoff_s'] for p in primitive if p['runoff_s'] is not None])),
        grasp_mark_s=float(np.median([p['starts']['Lifting and checking grip'] for p in pure])),
        lift_mark_s=float(np.median([p['starts']['Pouring from gripper'] for p in pure])),
        warp=np.mean([p['warp'] for p in primitive],axis=0).tolist(),grip=np.mean([p['grip'] for p in primitive],axis=0).tolist(),
        source_episodes=[p['path'] for p in primitive])
    fitted['warp'][0]=0.;fitted['warp'][-1]=1.;fitted['grip'][-1]=0.
    (ROOT/'primitive-fit.json').write_text(json.dumps(fitted,indent=2)+'\n')
    (ROOT/'motion-manifest.json').write_text(json.dumps({s:[dict(name=e['name'],path=e['path'],frames=len(e['q'])) for e in es] for s,es in result.items()},indent=2)+'\n')
    return result,fitted
