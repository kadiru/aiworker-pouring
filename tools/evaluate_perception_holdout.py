"""Once-only CPU scoring of the nine component-selected perception holdouts.

Uses actual image localization and temporal confirmation. Native fixture poses
and measured surfaces are scoring labels only; no fitting or threshold edits.
"""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_models import CupEstimator
from pour_guard_vision import locate,plane_point,cup_crop,surface_features,color_geometry_readout
from pour_perception_geometry import settled_label
from pour_perception_study import check_freeze
from pour_temporal_localization import consensus
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    output=ROOT/'perception-holdout.json';assert not output.exists()
    selection=read(ROOT/'component-selection.json');manifest=read(ROOT/'candidate.json')
    check_freeze(selection,manifest)
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete']
    plan=read(ROOT/'plan.json');gate=plan['component_gates'];expected={s['name']:s for s in plan['perception_holdout']}
    indexed={r['name']:r for r in audit['records']};assert expected.keys()<=indexed.keys()
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);torch.set_num_threads(4)
    model=CupEstimator().eval();model.load_state_dict(saved['cup_model']);results=[]
    for name,spec in expected.items():
        record=indexed[name];path=Path(record['path']);assert sha(path)==record['sha256']
        actual=read(path.parent/'parameters.json');assert all(actual[k]==v for k,v in spec.items())
        label=settled_label(read(path.parent/'samples.json'))
        localization=copy.deepcopy(saved['cup_localization_calibration']);detail=copy.deepcopy(saved['cup_detail_calibration'])
        detail['cup_geometry']=spec['declared_cup_geometry'];history=[];accepted=None;estimates=[]
        with h5py.File(path) as f:
            d=f['data/demo_0'];times=d['time'][:]
            for i,t in enumerate(times):
                if t-times[0]>saved['localization_consensus']['timeout_s']:break
                rgb=decode(d['obs/cup_detail_image'][i],'jpeg95')
                resized=np.asarray(Image.fromarray(rgb).resize((384,384),Image.Resampling.BILINEAR)).copy()
                uv,score=locate(resized,saved['cup_bank'],device='cpu');history.append((uv.tolist(),score))
                accepted=consensus(history,saved['thresholds']['cup_localization_score'],saved['localization_consensus']['cup'])
                if accepted is not None:break
            xy=None;localization_error=None
            if accepted is not None and label['accepted']:
                xy=plane_point(accepted['uv'],detail['table_z']+detail['cup_geometry']['rim_z_m'],localization)[:2]
                localization_error=float(np.linalg.norm(xy-np.asarray(spec['cup_world'][:2]))*1000)
                next_times=d['next_time'][:]
                for i in np.flatnonzero(next_times>=next_times[-1]-1.-1e-7):
                    rgb=decode(d['next_obs/cup_detail_image'][i],'jpeg95')
                    crop=torch.from_numpy(cup_crop(rgb,xy,detail))[None]
                    cnn=float((model(crop)[0]+saved['cup_bias']).clamp(0,1));features,_,mass=surface_features(crop)
                    geometry=saved['surface_geometry'];geo=float(color_geometry_readout(crop,geometry)[0]) if mass[0]>1 else 0.
                    estimates.append(dict(index=int(i),cnn=cnn,geometry=geo,error=cnn-label['target'],geometry_error=geo-label['target']))
        results.append(dict(name=name,source_sha256=record['sha256'],height_scale=spec['cup_height_scale'],diameter_scale=spec['cup_diameter_scale'],
            measured_label=label,temporal_confirmation=accepted,frames_considered=len(history),localization_error_mm=localization_error,
            localized_xy=None if xy is None else xy.tolist(),estimates=estimates))
    errors=[abs(e['error']) for r in results for e in r['estimates']]
    near=[e for r in results if r['measured_label']['target']>.85 for e in r['estimates']]
    metrics=dict(cases=len(results),accepted_labels=sum(r['measured_label']['accepted'] for r in results),
        confirmed_cups=sum(r['temporal_confirmation'] is not None for r in results),
        max_localization_error_mm=max((r['localization_error_mm'] for r in results if r['localization_error_mm'] is not None),default=None),
        level_mae=None if not errors else float(np.mean(errors)),level_max_abs=max(errors,default=None),
        near_full_frames=len(near),near_full_max_abs=max((abs(e['error']) for e in near),default=None),
        near_full_max_overestimate=max((e['error'] for e in near),default=None),
        geometry_max_abs=max((abs(e['geometry_error']) for r in results for e in r['estimates']),default=None),
        min_target=min(r['measured_label']['target'] for r in results),max_target=max(r['measured_label']['target'] for r in results))
    passed=bool(metrics['accepted_labels']==metrics['confirmed_cups']==len(expected)
        and metrics['max_localization_error_mm']<gate['localization_max_error_mm']
        and metrics['level_mae']<gate['level_validation_mae']
        and metrics['geometry_max_abs']<gate['geometry_validation_max_error']
        and metrics['near_full_frames']>=20 and metrics['max_target']>=.94 and metrics['min_target']<=.15
        and metrics['near_full_max_abs']<gate['level_validation_near_max_error']
        and metrics['near_full_max_overestimate']<gate['level_validation_near_overestimate'])
    report=dict(passed=passed,metrics=metrics,cases=results,checkpoint_sha256=manifest['checkpoint_sha256'],plan_sha256=manifest['plan_sha256'],
        component_selection_sha256=sha(ROOT/'component-selection.json'),audit_sha256=sha(ROOT/'recording-audit.json'),helper_sha256=sha(Path(__file__)),
        scope='Nine once-only static native original-cup holdouts with small position jitters. Known original size, image-localized position. Low and near-full measured coverage required. Resized autonomy remains gated. No autonomous grasp/pour claim and no fitting or threshold selection.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=passed,metrics=metrics)))
if __name__=='__main__':main()
