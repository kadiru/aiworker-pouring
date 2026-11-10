"""Combine reviewed static data and measured moving views, by whole episode.

Retired static test images never enter this loader. Camera crops use frozen RGB
localization; simulator coordinates are used only to measure localization error.
"""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/src','/workspace']
import fit_scale_level as baseline
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1');baseline.ROOT=ROOT
from fit_scale_level import sha,read
from validate_pour_recording import h5py,decode
from pour_guard_vision import cup_crop,plane_point
from pour_temporal_localization import consensus
from scale_rim_refiner import locate_static
from scale_data_admission import balanced_frame_weights

@torch.inference_mode()
def collect(parts):
    plan,parent,_,data=baseline.collect();protocol=read(ROOT/'protocol.json')
    assert not protocol.get('draft_only')
    specs={s['name']:s for s in plan['collect']};windows=[]
    loc=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False)
    resolution=tuple(loc['calibration']['resolution_wh']);new={k:dict(images=[],target=[],xy=[],meta=[]) for k in data}
    for part in parts:
        assert part in ('seed_pilot','seed_remaining','low_train','train','validation')
        audit_path=ROOT/(part+'-r0-audit.json');audit=read(audit_path);review=read(ROOT/(part+'-r0-manual-review.json'))
        assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(audit_path)
        declared=protocol.get(part)
        if part.startswith('seed_'):
            pilot=set(protocol['seed_pilot_names']);declared=[s for s in protocol['seed'] if (s['name'] in pilot)==(part=='seed_pilot')]
        own={s['name']:s for s in declared};assert own.keys()=={r['name'] for r in audit['records']}
        for r in audit['records']:
            s=own[r['name']];assert s['name'] not in specs;specs[s['name']]=s
            path=Path(r['path']);assert sha(path)==r['sha256'] and r['recording_validation']['valid'] and r['label']['accepted']
            recorded=read(path.parent/'parameters.json');assert all(recorded[k]==v for k,v in s.items())
            assert r['split']==s['study_split'] and r['split'] in new
            if s.get('calibration_motion'):assert r['motion']['passed'] and r['motion']['cup_inventory_range_ml']<=.005833
            with h5py.File(path) as f:
                d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,cup_geometry=s['declared_cup_geometry'])
                k=np.asarray(cal['K']).copy();k[:2]*=resolution[0]/1024
                assert np.allclose(k,loc['calibration']['K']) and np.allclose(cal['T'],loc['calibration']['T'])
                history=[];accepted=None
                for i in range(12):
                    image=decode(d['obs/cup_detail_image'][i],'jpeg95');image=np.asarray(Image.fromarray(image).resize(resolution,Image.Resampling.BILINEAR)).copy()
                    uv,score=locate_static(image,loc['bank']);history.append((uv.tolist(),score));accepted=consensus(history,loc['threshold'],loc['consensus'])
                    if accepted is not None:break
                xy=None if accepted is None else plane_point(accepted['uv'],1.04+s['declared_cup_geometry']['rim_z_m'],loc['calibration'])[:2]
                error=None if xy is None else float(np.linalg.norm(xy-s['cup_world'][:2])*1000)
                passed=error is not None and error<protocol['gates']['localization_max_error_mm']
                windows.append(dict(name=s['name'],confirmed=accepted is not None,error_mm=error,frames=len(history),passed=passed))
                if not passed:continue
                times=d['next_time'][:];moving=s.get('calibration_motion')
                indices=np.flatnonzero((times>=.5)&(times<=2.5)) if moving else np.flatnonzero(times>=times[-1]-1.-1e-7)
                samples=read(path.parent/'samples.json');sample_times=np.asarray([v['t'] for v in samples]);a=new[r['split']]
                for index in indices[::2]:
                    rgb=decode(d['next_obs/cup_detail_image'][int(index)],'jpeg95');a['images'].append(cup_crop(rgb,xy,cal));a['xy'].append(xy)
                    label=r['label']['target']
                    if moving:
                        j=int(np.abs(sample_times-times[index]).argmin());assert abs(sample_times[j]-times[index])<1e-6
                        label=samples[j]['cup_fill_fraction']
                    a['target'].append(label);a['meta'].append(dict(name=s['name'],cup_height_scale=s['cup_height_scale'],cup_diameter_scale=s['cup_diameter_scale'],index=int(index),source_sha256=r['sha256'],moving=bool(moving)))
    report=dict(passed=all(w['passed'] for w in windows),windows=windows,parts=parts,protocol_sha256=sha(ROOT/'protocol.json'),locator_sha256=sha(ROOT/'cup-neural-localization.pt'))
    (ROOT/'dynamic-localization-check.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed'],'New view localization gate failed; no level/color fitting permitted'
    for split,a in data.items():
        extra=new[split]
        if extra['images']:
            a['x']=torch.cat((a['x'],torch.tensor(np.stack(extra['images']),device='cuda')))
            a['xy']=torch.cat((a['xy'],torch.tensor(np.asarray(extra['xy']),device='cuda',dtype=torch.float32)))
            a['target']+=extra['target'];a['meta']+=extra['meta'];a['y']=torch.tensor(a['target'],device='cuda')
        a['weights']=torch.tensor(balanced_frame_weights(a['meta']),device='cuda');a['geometry']=[specs[r['name']]['declared_cup_geometry'] for r in a['meta']]
    return protocol,parent,data
