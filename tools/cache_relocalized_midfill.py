"""Rebuild every level crop from accepted image localization, never true pose."""
import json,time
from pathlib import Path
import numpy as np
import torch
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from perception_midfill_data import entries
from dynamic_cup_tracker import CupTracker
from validate_pour_recording import h5py,decode
from pour_guard_vision import cup_crop,plane_point

@torch.inference_mode()
def main():
    assert not (ROOT/'level-cache-v6.json').exists() and not (ROOT/'component-selection.json').exists()
    assert read(ROOT/'tracker-t3.json')['passed']
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4);started=time.monotonic();protected_unchanged()
    loc=torch.load(ROOT/'tracker-t3.pt',map_location='cpu',weights_only=False);assert loc['accepted']
    data={s:dict(x=[],xy=[],target=[],meta=[],geometry=[]) for s in ('train','validation')};admission=[]
    for spec,record,split in entries():
        path=Path(record['path']);assert sha(path)==record['sha256']
        assert all(read(path.parent/'parameters.json')[k]==v for k,v in spec.items())
        tracker=CupTracker(loc,spec['declared_cup_geometry']);samples=read(path.parent/'samples.json')
        sample_times=np.asarray([v['t'] for v in samples]);moving=bool(spec.get('calibration_motion'))
        with h5py.File(path) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,cup_geometry=spec['declared_cup_geometry'])
            assert np.allclose(cal['K'],loc['calibration']['K']) and np.allclose(cal['T'],loc['calibration']['T'])
            state=None
            for i in range(12):
                state=tracker.update(decode(d['obs/cup_detail_image'][i],'jpeg95'))
                if state['valid']:break
            assert state['valid']
            xy=plane_point(state['uv'],cal['table_z']+spec['declared_cup_geometry']['rim_z_m'],loc['calibration'])[:2]
            error=float(np.linalg.norm(xy-spec['cup_world'][:2])*1000)
            assert error<read(ROOT/'protocol.json')['gates']['localization_max_error_mm']
            times=d['next_time'][:]
            indices=np.flatnonzero((times>=.5)&(times<=2.5)) if moving else np.flatnonzero(times>=times[-1]-1.-1e-7)[::2]
            assert len(indices)==(40 if moving else 11)
            a=data[split]
            for index in indices:
                image=decode(d['next_obs/cup_detail_image'][int(index)],'jpeg95');assert tracker.update(image)['valid']
                target=record['label']['target']
                if moving:
                    j=int(np.abs(sample_times-times[index]).argmin());assert abs(sample_times[j]-times[index])<1e-6
                    target=samples[j]['cup_fill_fraction']
                a['x'].append(cup_crop(image,xy,cal));a['xy'].append(xy);a['target'].append(target)
                a['geometry'].append(spec['declared_cup_geometry'])
                a['meta'].append(dict(name=spec['name'],cup_height_scale=spec['cup_height_scale'],cup_diameter_scale=spec['cup_diameter_scale'],index=int(index),source_sha256=record['sha256'],moving=moving))
            admission.append(dict(name=spec['name'],split=split,error_mm=error,frames=len(indices),source_sha256=record['sha256']))
    arrays={};metadata={}
    for split,a in data.items():
        arrays[split+'_x']=np.stack(a.pop('x'));arrays[split+'_xy']=np.asarray(a.pop('xy'),dtype=np.float32)
        arrays[split+'_y']=np.asarray(a['target'],dtype=np.float32);metadata[split]=a
    np.savez_compressed(ROOT/'level-cache-v6.npz',**arrays)
    (ROOT/'level-cache-v6-meta.json').write_text(json.dumps(metadata)+'\n')
    result=dict(passed=True,tracker_sha256=sha(ROOT/'tracker-t3.pt'),tracker_module_sha256=sha(ROOT/'dynamic_cup_tracker.py'),
                protocol_sha256=sha(ROOT/'protocol.json'),cache_sha256=sha(ROOT/'level-cache-v6.npz'),metadata_sha256=sha(ROOT/'level-cache-v6-meta.json'),
                counts={s:dict(episodes=len({m['name'] for m in a['meta']}),frames=len(a['target'])) for s,a in metadata.items()},
                admissions=admission,total_seconds=time.monotonic()-started,helper_sha256=sha(Path(__file__)))
    protected_unchanged()
    (ROOT/'level-cache-v6.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts']),flush=True)

if __name__=='__main__':main()
