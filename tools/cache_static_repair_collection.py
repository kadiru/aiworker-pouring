"""Append reviewed new training frames to the immutable original crop cache."""
import json,time
from pathlib import Path
import numpy as np
import torch
from dynamic_level_data import ROOT,read,sha
from dynamic_cup_tracker import CupTracker
from validate_pour_recording import h5py,decode
from pour_guard_vision import cup_crop,plane_point

def main():
    assert not (ROOT/'level-cache-v3.json').exists() and not (ROOT/'component-selection.json').exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);started=time.monotonic()
    old=read(ROOT/'level-cache-v1.json');p=read(ROOT/'protocol.json');m=read(ROOT/'candidate.json')
    assert old['cache_sha256']==sha(ROOT/'level-cache-v1.npz') and old['metadata_sha256']==sha(ROOT/'level-cache-v1-meta.json')
    assert p['version']==5 and old['tracker_sha256']==sha(ROOT/'tracker-t1.pt') and old['tracker_module_sha256']==sha(ROOT/'dynamic_cup_tracker.py')
    assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    audit=read(ROOT/'train-r2-audit.json');review=read(ROOT/'train-r2-manual-review.json')
    assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ROOT/'train-r2-audit.json')
    specs={s['name']:s for s in p['train'] if s['name'] in p['repair_train_names']}
    assert specs.keys()=={r['name'] for r in audit['records']}
    loc=torch.load(ROOT/'tracker-t1.pt',map_location='cpu',weights_only=False);assert loc['accepted']
    arrays=dict(np.load(ROOT/'level-cache-v1.npz'));metadata=read(ROOT/'level-cache-v1-meta.json')
    images=[];xy_values=[];targets=[];extra_meta=[];geometries=[];windows=[]
    for r in audit['records']:
        s=specs[r['name']];path=Path(r['path']);samples=read(path.parent/'samples.json');sample_times=np.asarray([v['t'] for v in samples])
        assert s['study_split']=='train' and r['split']=='train' and sha(path)==r['sha256']
        assert r['recording_validation']['valid'] and r['label']['accepted'] and all(v==0 for v in r['containment'].values())
        assert all(read(path.parent/'parameters.json')[k]==v for k,v in s.items())
        tracker=CupTracker(loc,s['declared_cup_geometry']);raw_errors=[];result=None
        with h5py.File(path) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,cup_geometry=s['declared_cup_geometry'])
            assert np.allclose(cal['K'],loc['calibration']['K']) and np.allclose(cal['T'],loc['calibration']['T'])
            z=1.04+s['declared_cup_geometry']['rim_z_m']
            for i in range(12):
                image=decode(d['obs/cup_detail_image'][i],'jpeg95');uv,_=tracker.detect(image,refine=True)
                raw_errors.append(float(np.linalg.norm(plane_point(uv,z,loc['calibration'])[:2]-s['cup_world'][:2])*1000))
            for i in range(12):
                result=tracker.update(decode(d['obs/cup_detail_image'][i],'jpeg95'))
                if result['valid']:break
            confirmed=result['valid'];xy=plane_point(result['uv'],z,loc['calibration'])[:2] if confirmed else None
            err=float(np.linalg.norm(xy-s['cup_world'][:2])*1000) if confirmed else None
            frames=[];times=d['next_time'][:];indices=np.flatnonzero(times>=times[-1]-1.-1e-7)[::2];assert len(indices)==11
            if confirmed:
                for index in indices:
                    image=decode(d['next_obs/cup_detail_image'][int(index)],'jpeg95');state=tracker.update(image)
                    frames.append(dict(index=int(index),valid=state['valid'],reason=state['reason']))
                    j=int(np.abs(sample_times-times[index]).argmin());assert abs(sample_times[j]-times[index])<1e-6
                    images.append(cup_crop(image,xy,cal));xy_values.append(xy);targets.append(r['label']['target']);geometries.append(s['declared_cup_geometry'])
                    extra_meta.append(dict(name=s['name'],cup_height_scale=s['cup_height_scale'],cup_diameter_scale=s['cup_diameter_scale'],index=int(index),source_sha256=r['sha256'],moving=False))
            passed=confirmed and err<p['gates']['localization_max_error_mm'] and max(raw_errors)<p['gates']['localization_max_error_mm'] and len(frames)==11 and all(v['valid'] for v in frames)
            windows.append(dict(name=s['name'],passed=passed,confirmed=confirmed,error_mm=err,raw_max_error_mm=max(raw_errors),frames=frames))
            print(json.dumps(dict(name=s['name'],passed=passed,error_mm=err,raw_max=max(raw_errors))),flush=True)
    admission=dict(passed=all(w['passed'] for w in windows),cases=windows,tracker_sha256=sha(ROOT/'tracker-t1.pt'),protocol_sha256=sha(ROOT/'protocol.json'),all_frames_required=True)
    (ROOT/'static-repair-tracker-admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    assert admission['passed'],'Frozen tracker must admit all new scenes; no failed frames may be discarded'
    arrays['train_x']=np.concatenate((arrays['train_x'],np.stack(images)))
    arrays['train_xy']=np.concatenate((arrays['train_xy'],np.asarray(xy_values,dtype=np.float32)))
    arrays['train_y']=np.concatenate((arrays['train_y'],np.asarray(targets,dtype=np.float32)))
    metadata['train']['meta']+=extra_meta;metadata['train']['target']+=targets;metadata['train']['geometry']+=geometries
    np.savez_compressed(ROOT/'level-cache-v3.npz',**arrays)
    (ROOT/'level-cache-v3-meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    final=dict(passed=True,counts={split:dict(episodes=len({v['name'] for v in b['meta']}),frames=len(b['meta'])) for split,b in metadata.items()},
        parent_cache_sha256=old['cache_sha256'],cache_sha256=sha(ROOT/'level-cache-v3.npz'),metadata_sha256=sha(ROOT/'level-cache-v3-meta.json'),tracker_sha256=old['tracker_sha256'],tracker_module_sha256=old['tracker_module_sha256'],admission_sha256=sha(ROOT/'static-repair-tracker-admission.json'),
        protocol_sha256=sha(ROOT/'protocol.json'),audit_sha256=sha(ROOT/'train-r2-audit.json'),review_sha256=sha(ROOT/'train-r2-manual-review.json'),helper_sha256=sha(Path(__file__)),total_seconds=time.monotonic()-started,
        scope='Appended 36 training-only native episodes; original train/validation crop bytes and splits unchanged, all reserved images excluded. Simulator positions only score admission; measured levels only supply labels.')
    (ROOT/'level-cache-v3.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final))
if __name__=='__main__':main()
