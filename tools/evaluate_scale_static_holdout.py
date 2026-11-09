"""Once-only CPU scoring of frozen static components on nine unused scenes."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_models import CupEstimator
from pour_guard_vision import locate,plane_point,cup_crop,color_geometry_readout
from pour_temporal_localization import consensus
from pour_settled_model import features,predict
ROOT=Path('/workspace/learning/A-scale-20261109-v1')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    output=ROOT/'static-holdout-evaluation.json';assert not output.exists()
    selection=read(ROOT/'component-selection.json');m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json')
    assert selection['all_size_component_gates_passed'] and not selection['autonomous_use_accepted']
    assert selection['evaluation_helper_sha256']==sha(Path(__file__)) and selection['evaluation_observation_frames']==12
    assert all(selection[k]==m[k] for k in ('checkpoint_sha256','plan_sha256','runtime_sources'))
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(Path(r['path']))==r['sha256'] for r in selection['components'].values())
    audit=read(ROOT/'holdout-r0-audit.json');assert audit['complete'] and audit['plan_sha256']==m['plan_sha256']
    indexed={r['name']:r for r in audit['records']};expected={s['name']:s for s in plan['holdout']}
    assert set(indexed)==set(expected)==set(selection['holdout_names'])
    loc=torch.load(selection['components']['cup-neural-localization.pt']['path'],map_location='cpu',weights_only=False)
    level=torch.load(selection['components']['static-level-candidate.pt']['path'],map_location='cpu',weights_only=False)
    assert level['accepted'];torch.set_num_threads(2);model=CupEstimator().eval();model.load_state_dict(level['cup_model'])
    parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    parent_model=CupEstimator().eval();parent_model.load_state_dict(parent['cup_model'])
    rows=[];gate=plan['component_gates']
    for name,s in expected.items():
        r=indexed[name];path=Path(r['path']);assert sha(path)==r['sha256'] and r['recording_validation']['valid']
        actual=read(path.parent/'parameters.json');assert all(actual[k]==v for k,v in s.items())
        history=[];accepted=None;estimates=[];error_mm=None;parent_history=[];parent_accepted=None;parent_error_mm=None;parent_estimates=[]
        with h5py.File(path) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,cup_geometry=s['declared_cup_geometry'])
            scaled=np.asarray(cal['K']).copy();scaled[:2]*=384/1024
            assert np.allclose(scaled,loc['calibration']['K']) and np.allclose(cal['T'],loc['calibration']['T'])
            for i in range(min(12,len(d['time']))):
                image=decode(d['obs/cup_detail_image'][i],'jpeg95')
                image=np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy()
                uv,score=locate(image,loc['bank'],device='cpu');history.append((uv.tolist(),score))
                accepted=consensus(history,loc['threshold'],loc['consensus'])
                if accepted is not None:break
            # Same unused scene and fixed twelve-frame budget, evaluated with
            # the frozen parent only as a reference. It cannot replace the
            # selected candidate or change any primary pass/fail outcome.
            for i in range(min(12,len(d['time']))):
                image=decode(d['obs/cup_detail_image'][i],'jpeg95')
                image=np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy()
                uv,score=locate(image,parent['cup_bank'],device='cpu');parent_history.append((uv.tolist(),score))
                parent_accepted=consensus(parent_history,parent['thresholds']['cup_localization_score'],parent['localization_consensus']['cup'])
                if parent_accepted is not None:break
            if accepted is not None and r['label']['accepted']:
                xy=plane_point(accepted['uv'],1.04+s['declared_cup_geometry']['rim_z_m'],loc['calibration'])[:2]
                error_mm=float(np.linalg.norm(xy-np.asarray(s['cup_world'][:2]))*1000)
                times=d['next_time'][:]
                for index in np.flatnonzero(times>=times[-1]-1.-1e-7):
                    image=decode(d['next_obs/cup_detail_image'][int(index)],'jpeg95')
                    x=torch.from_numpy(cup_crop(image,xy,cal))[None]
                    cnn=(model(x)+level['cup_bias']).clamp(0,1)
                    estimate=float(predict(features(x,cnn,xy),level['settled_regressor'])[0])
                    color=float(color_geometry_readout(x,level['surface_geometry'],xy)[0])
                    estimates.append(dict(index=int(index),level=estimate,error=estimate-r['label']['target'],
                        raw_cnn=float(cnn[0]),independent_color=color,color_error=color-r['label']['target']))
            if parent_accepted is not None and r['label']['accepted']:
                pc=parent['cup_localization_calibration'];pd=copy.deepcopy(parent['cup_detail_calibration'])
                pd['cup_geometry']=s['declared_cup_geometry']
                assert np.allclose(pc['K'],loc['calibration']['K']) and np.allclose(pc['T'],loc['calibration']['T'])
                assert np.allclose(pd['K'],cal['K']) and np.allclose(pd['T'],cal['T'])
                xy=plane_point(parent_accepted['uv'],1.04+s['declared_cup_geometry']['rim_z_m'],pc)[:2]
                parent_error_mm=float(np.linalg.norm(xy-np.asarray(s['cup_world'][:2]))*1000)
                times=d['next_time'][:]
                for index in np.flatnonzero(times>=times[-1]-1.-1e-7):
                    image=decode(d['next_obs/cup_detail_image'][int(index)],'jpeg95')
                    x=torch.from_numpy(cup_crop(image,xy,pd))[None]
                    cnn=(parent_model(x)+parent['cup_bias']).clamp(0,1)
                    estimate=float(predict(features(x,cnn,xy),parent['settled_regressor'])[0])
                    parent_estimates.append(dict(index=int(index),level=estimate,error=estimate-r['label']['target']))
        errors=np.asarray([v['error'] for v in estimates])
        case_pass=bool(r['label']['accepted'] and error_mm is not None and error_mm<gate['localization_max_error_mm']
            and len(errors)>=20 and np.abs(errors).mean()<gate['level_validation_mae']
            and np.abs(errors).max()<gate['level_validation_near_max_error'] and errors.max()<gate['level_validation_near_overestimate'])
        rows.append(dict(name=name,height_scale=s['cup_height_scale'],diameter_scale=s['cup_diameter_scale'],
            source_sha256=r['sha256'],label=r['label'],temporal_confirmation=accepted,
            localization_error_mm=error_mm,frames_considered=len(history),estimates=estimates,passed=case_pass,
            parent_reference=dict(confirmation=parent_accepted,localization_error_mm=parent_error_mm,
                                  frames_considered=len(parent_history),estimates=parent_estimates)))
    errors=[e['error'] for r in rows for e in r['estimates']]
    metrics=dict(cases=len(rows),passed=sum(r['passed'] for r in rows),confirmed=sum(r['temporal_confirmation'] is not None for r in rows),
        max_localization_error_mm=max((r['localization_error_mm'] for r in rows if r['localization_error_mm'] is not None),default=None),
        level_mae=None if not errors else float(np.abs(errors).mean()),level_max_abs=max(map(abs,errors),default=None),
        level_max_overestimate=max(errors,default=None),min_label=min(r['label']['target'] for r in rows),max_label=max(r['label']['target'] for r in rows),
        independent_color_max_abs=max((abs(e['color_error']) for r in rows for e in r['estimates']),default=None))
    parent_errors=[e['error'] for r in rows for e in r['parent_reference']['estimates']]
    parent_reference=dict(confirmed=sum(r['parent_reference']['confirmation'] is not None for r in rows),
        max_localization_error_mm=max((r['parent_reference']['localization_error_mm'] for r in rows if r['parent_reference']['localization_error_mm'] is not None),default=None),
        level_mae=None if not parent_errors else float(np.abs(parent_errors).mean()),
        level_max_abs=max(map(abs,parent_errors),default=None),
        scope='Predeclared reference only, same twelve-frame localization budget and same declared geometry. Level metrics include confirmed parent detections, even when their positions are inaccurate. No model selection from these results.')
    report=dict(passed=all(r['passed'] for r in rows),metrics=metrics,cases=rows,parent_reference=parent_reference,
        component_selection_sha256=sha(ROOT/'component-selection.json'),audit_sha256=sha(ROOT/'holdout-r0-audit.json'),
        plan_sha256=m['plan_sha256'],helper_sha256=sha(Path(__file__)),autonomous_use_accepted=False,
        scope='Once-only independent near-full static observations at nine declared sizes and small position jitters. No fitting, retries, unknown-size inference or autonomous task claim. Higher-fill, low-fill independent holdouts, held-bottle/dynamic scenes and independent color acceptance remain separate requirements.')
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=report['passed'],metrics=metrics)),flush=True)


if __name__=='__main__':main()
