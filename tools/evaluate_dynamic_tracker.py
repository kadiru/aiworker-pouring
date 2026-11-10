"""Causal development tracker evaluation, with explicit rejection challenges."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from dynamic_tracker_data import entries
from dynamic_cup_tracker import CupTracker
from validate_pour_recording import h5py,decode
from pour_guard_vision import plane_point,project

def error(uv,s,cal):
    return float(np.linalg.norm(plane_point(uv,cal['table_z']+s['declared_cup_geometry']['rim_z_m'],cal)[:2]-s['cup_world'][:2])*1000)

def challenges(tracker,image,s):
    baseline=copy.copy(tracker);baseline.failed=False;normal=baseline.update(image)
    assert normal['valid'],'Challenge source must itself be accepted'
    cal=tracker.artifact['calibration'];z=cal['table_z']+s['declared_cup_geometry']['rim_z_m']
    # Labels/geometry only author synthetic challenge images, never inference inputs.
    truth=project(np.r_[s['cup_world'][:2],z],cal);px,py=np.rint(truth*384/1024).astype(int)
    negative=image.copy();x0,y0=np.floor(np.array([px-50,py-24])*1024/384).astype(int);x1,y1=np.ceil(np.array([px+50,py+48])*1024/384).astype(int)
    negative[max(0,y0):min(1024,y1),max(0,x0):min(1024,x1)]=180
    variants=[('cup_erased',negative)]
    base=np.r_[s['cup_world'][:2],cal['table_z']+s['declared_cup_geometry']['bottom_z_m']]
    for dx,dy in ((.0035,0),(-.0035,0),(0,.0035),(0,-.0035)):
        delta=np.rint(project(base+[dx,dy,0],cal)-project(base,cal)).astype(int)
        shifted=np.asarray(Image.fromarray(image).transform((1024,1024),Image.Transform.AFFINE,(1,0,-int(delta[0]),0,1,-int(delta[1])),Image.Resampling.NEAREST,fillcolor=(180,180,180))).copy()
        variants.append((f'view_shift_{dx}_{dy}',shifted))
    rows=[]
    for name,im in variants:
        probe=copy.copy(tracker);probe.failed=False;r=probe.update(im)
        rows.append(dict(kind=name,rejected=not r['valid'],reason=r['reason'],movement_mm=r.get('movement_mm'),similarity=r.get('similarity')))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--round',choices=('t0','t1'),default='t0');args=p.parse_args();prefix='tracker-'+args.round
    cfg=read(ROOT/(prefix+'-protocol.json'));assert not (ROOT/(prefix+'.json')).exists() and not (ROOT/'component-selection.json').exists()
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items());protected_unchanged()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);started=time.perf_counter()
    old=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False);new=torch.load(ROOT/'rim-d2.pt',map_location='cpu',weights_only=False)
    coarse=torch.load(ROOT/'coarse-d2.pt',map_location='cpu',weights_only=False)
    artifact=dict(accepted=False,bank=dict(kind='stationary_cup_tracker'),coarse_state=coarse['state'],fine_banks=[old['bank'],new['bank']],settings=cfg['settings'],calibration=new['calibration'],threshold=new['threshold'],consensus=new['consensus'])
    rows=[];cases=[];negative=[];parity=[]
    for s,record,split in entries():
        path=Path(record['path']);assert sha(path)==record['sha256'] and all(read(path.parent/'parameters.json')[k]==v for k,v in s.items())
        tracker=CupTracker(artifact,s['declared_cup_geometry']);cpu=CupTracker(artifact,s['declared_cup_geometry'],'cpu') if split=='validation' else None
        first_reference=None;initial=[];track=[];raw=[];parity_initial=[];last_image=None
        with h5py.File(path) as f:
            d=f['data/demo_0'];assert np.allclose(f['calibration/cup_detail/intrinsics'][:],artifact['calibration']['K']) and np.allclose(d['obs/cup_detail_world_T_camera'][0],artifact['calibration']['T'])
            frames=[('obs',i,float(d['time'][i])) for i in range(12)]+[('next_obs',len(d['next_time'])-1,float(d['next_time'][-1]))]
            if s.get('calibration_motion'):frames += [('next_obs',int(i),float(d['next_time'][i])) for i in np.flatnonzero((d['next_time'][:]>=.5)&(d['next_time'][:]<=2.5))]
            frames=sorted(frames,key=lambda v:(v[2],v[0],v[1]))
            for group,i,t in frames:
                image=decode(d[group+'/cup_detail_image'][i],'jpeg95');result=tracker.update(image);e=None if not result['valid'] else error(result['uv'],s,artifact['calibration'])
                if result['valid'] and first_reference is None:first_reference=(copy.copy(tracker),image.copy())
                if group=='obs':
                    point=result.get('raw_uv')
                    if point is None:point,_=tracker.detect(image,refine=True)
                    raw.append(error(point,s,artifact['calibration']));initial.append(result['valid'])
                    if cpu is not None:
                        c=cpu.update(image);same=c['valid']==result['valid'];delta=None if not c['valid'] or not result['valid'] else float(np.linalg.norm(np.asarray(c['uv'])-result['uv']))
                        parity_initial.append(dict(status_equal=same,pixels=delta))
                elif t>=.5:track.append(result['valid'])
                rows.append(dict(name=s['name'],split=split,group=group,index=i,t=t,error_mm=e,**result));last_image=image
            if cpu is not None:
                c=cpu.update(last_image);g=copy.copy(tracker).update(last_image)
                parity_initial.append(dict(status_equal=c['valid']==g['valid'],pixels=float(np.linalg.norm(np.asarray(c['uv'])-g['uv'])) if c['valid'] and g['valid'] else None))
                if first_reference is not None:
                    ref,image=first_reference;negative += [dict(name=s['name'],**v) for v in challenges(ref,image,s)]
                parity += [dict(name=s['name'],**v) for v in parity_initial]
        own=rows[-len(frames):];confirmed=any(initial);max_error=max((v['error_mm'] for v in own if v['error_mm'] is not None),default=None)
        passed=confirmed and all(track) and max_error is not None and max_error<3 and max(raw)<3 and all(initial[initial.index(True):])
        case=dict(name=s['name'],split=split,passed=passed,confirmed=confirmed,frames=len(frames),raw_initial_max_mm=max(raw),tracked_max_mm=max_error,rejections=sum(v['reason'] not in ('confirming','tracked') for v in own));cases.append(case)
        print(json.dumps(case),flush=True)
    parities=[v['pixels'] for v in parity if v['pixels'] is not None]
    passed=all(v['passed'] for v in cases) and len(negative)==36*5 and all(v['rejected'] for v in negative) and all(v['status_equal'] for v in parity) and max(parities)<.1
    artifact['accepted']=passed;torch.save(artifact,ROOT/(prefix+'.pt'))
    report=dict(passed=passed,cases=cases,rows=rows,synthetic_challenges=negative,cpu_gpu_parity=parity,cpu_gpu_max_pixels=max(parities),checkpoint_sha256=sha(ROOT/(prefix+'.pt')),protocol_sha256=sha(ROOT/(prefix+'-protocol.json')),optimizer_steps=0,total_seconds=time.perf_counter()-started,autonomy_accepted=False,scope=cfg['scope'],accuracy_policy=cfg['accuracy_policy'])
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items());protected_unchanged()
    (ROOT/(prefix+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=sum(v['passed'] for v in cases),total=len(cases),rejected_challenges=sum(v['rejected'] for v in negative),cpu_gpu_max_pixels=max(parities))),flush=True)
if __name__=='__main__':main()
