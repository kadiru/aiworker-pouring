"""Once-only frozen moving-view evaluation; no fitting or runtime activation."""
import hashlib,json,sys,math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import cup_crop,plane_point,project
from pour_temporal_localization import consensus
from pour_cup_locator import CupLocator,centers
from scale_rim_refiner import locate_static
from scale_level_cnn import predict_size_cnn
from scale_color_mask import predict_mask
from dynamic_cup_tracker import CupTracker
from ensemble_level_model import predict_ensemble_level
from landmark_level_model import predict_landmark_level
from landmark_level_model_v2 import predict_landmark_level as predict_conditioned_level
from landmark_level_model_v3 import predict_landmark_level as predict_evaluation_precision_level
from repair_level_model import predict_repair
from repair_level_model_v2 import predict_repair as predict_repair_v2
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def primary_prediction(image,xy,geometry,artifact):
    if artifact.get('kind')=='rgb_equal_ensemble_v1':return predict_ensemble_level(image,xy,geometry,artifact)[0]
    if artifact.get('kind')=='geometry_landmarks_v3':return predict_evaluation_precision_level(image,xy,geometry,artifact)[0]
    if artifact.get('kind')=='geometry_landmarks_v2':return predict_conditioned_level(image,xy,geometry,artifact)[0]
    if artifact.get('kind')=='geometry_landmarks_v1':return predict_landmark_level(image,xy,geometry,artifact)[0]
    if artifact.get('kind')=='crop_tolerant_outline_v2':return predict_repair_v2(image,xy,geometry,artifact)[0]
    if artifact.get('kind')=='crop_tolerant_outline_v1':return predict_repair(image,xy,geometry,artifact)[0]
    return predict_size_cnn(image,xy,geometry,artifact)
def color_prediction(image,xy,geometry,artifact):
    if artifact.get('kind')=='full_precision_color_landmarks_v1':return predict_ensemble_level(image,xy,geometry,artifact)
    if artifact.get('kind')=='geometry_landmarks_v3':return predict_evaluation_precision_level(image,xy,geometry,artifact)
    if artifact.get('kind')=='geometry_landmarks_v2':return predict_conditioned_level(image,xy,geometry,artifact)
    if artifact.get('kind')=='geometry_landmarks_v1':return predict_landmark_level(image,xy,geometry,artifact)
    if artifact.get('kind')=='crop_tolerant_outline_v2':return predict_repair_v2(image,xy,geometry,artifact)
    if artifact.get('kind')=='crop_tolerant_outline_v1':return predict_repair(image,xy,geometry,artifact)
    return predict_mask(image,xy,geometry,artifact)
def error_metrics(values,target):
    values=np.asarray(values);target=np.asarray(target)
    if not np.isfinite(values).all():return dict(valid=False,mae=None,max_abs=None,near_max=None,near_overestimate=None)
    error=values-target;near=target>.85
    return dict(valid=True,mae=float(np.abs(error).mean()),max_abs=float(np.abs(error).max()),near_max=float(np.abs(error[near]).max()) if near.any() else None,near_overestimate=float(error[near].max()) if near.any() else None)
def level_ok(m,g):
    return m['valid'] and m['mae']<g['level_validation_mae'] and (m['near_max'] is None or (m['near_max']<g['level_validation_near_max_error'] and m['near_overestimate']<g['level_validation_near_overestimate']))
def localization_ok(frames,limit,threshold):
    return len(frames)==40 and all(v.get('localization_valid') and v.get('localization_error_mm') is not None and math.isfinite(v['localization_error_mm']) and v['localization_error_mm']<limit and math.isfinite(v['score']) and v['score']>=threshold for v in frames)

@torch.inference_mode()
def main():
    assert not (ROOT/'moving-holdout-evaluation.json').exists();selection=read(ROOT/'component-selection.json');p=read(ROOT/'protocol.json');g=p['gates'];m=read(ROOT/'candidate.json')
    assert selection['all_required_development_gates_passed'] and selection['protocol_sha256']==sha(ROOT/'protocol.json')
    assert selection['evaluation_helper_sha256']==sha(Path(__file__)) and all(sha(ROOT/n)==v for n,v in selection['frozen_files'].items())
    assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    audit=read(ROOT/'holdout-r0-audit.json');review=read(ROOT/'holdout-r0-manual-review.json');assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ROOT/'holdout-r0-audit.json')
    specs={s['name']:s for s in p['holdout']};records={r['name']:r for r in audit['records']};assert specs.keys()==records.keys()==set(selection['holdout_names'])
    loc=torch.load(ROOT/selection['locator'],map_location='cpu',weights_only=False);rgb=torch.load(ROOT/selection['rgb'],map_location='cpu',weights_only=False);color=torch.load(ROOT/selection['color'],map_location='cpu',weights_only=False)
    assert all(a['accepted'] for a in (loc,rgb,color));tracking=loc['bank']['kind']=='stationary_cup_tracker'
    coarse=CupLocator().cuda().eval();coarse.load_state_dict(loc['coarse_state'] if tracking else loc['bank']['coarse_state']);rows=[]
    for name,s in specs.items():
        r=records[name];path=Path(r['path']);assert sha(path)==r['sha256'] and all(read(path.parent/'parameters.json')[k]==v for k,v in s.items())
        physical=r['recording_validation']['valid'] and r['label']['accepted'] and r['motion']['passed'] and r['motion']['cup_inventory_range_ml']==0 and all(v==0 for v in r['containment'].values())
        if s['coverage_role'].endswith('high'):physical=physical and g['high_fill_min']<=r['label']['target']<=g['high_fill_max']
        history=[];anchor=None;frames=[];negative_scores=[];parity=[];status_parity=True;initial_raw_errors=[]
        tracker=CupTracker(loc,s['declared_cup_geometry']) if tracking else None
        cpu_tracker=CupTracker(loc,s['declared_cup_geometry'],'cpu') if tracking else None
        with h5py.File(path) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,cup_geometry=s['declared_cup_geometry'])
            assert np.allclose(cal['K'],loc['calibration']['K']) and np.allclose(cal['T'],loc['calibration']['T']);z=1.04+s['declared_cup_geometry']['rim_z_m']
            if tracking:
                # Pure raw predictions for scoring only: never advance tracker history with future frames.
                for i in range(12):
                    image=decode(d['obs/cup_detail_image'][i],'jpeg95');uv,_=tracker.detect(image,refine=True)
                    initial_raw_errors.append(float(np.linalg.norm(plane_point(uv,z,loc['calibration'])[:2]-s['cup_world'][:2])*1000))
            for i in range(12):
                image=decode(d['obs/cup_detail_image'][i],'jpeg95')
                if tracking:
                    tr=tracker.update(image);cr=cpu_tracker.update(image);history.append(tr);status_parity=status_parity and tr['valid']==cr['valid'];anchor=dict(uv=tr['uv']) if tr['valid'] else None
                    if tr['valid'] and cr['valid']:parity.append(float(np.linalg.norm(np.asarray(tr['uv'])-cr['uv'])))
                else:
                    uv,score=locate_static(image,loc['bank']);history.append((uv.tolist(),score));anchor=consensus(history,loc['threshold'],loc['consensus'])
                if anchor is not None:break
            anchor_error=None;xy=None
            if anchor is not None:
                xy=plane_point(anchor['uv'],z,loc['calibration'])[:2];anchor_error=float(np.linalg.norm(xy-s['cup_world'][:2])*1000)
            samples=read(path.parent/'samples.json');sample_times=np.asarray([v['t'] for v in samples]);times=d['next_time'][:];indices=np.flatnonzero((times>=.5)&(times<=2.5));assert len(indices)==40
            if xy is not None:
                for i in indices:
                    image=decode(d['next_obs/cup_detail_image'][int(i)],'jpeg95')
                    if tracking:
                        tr=tracker.update(image);cr=cpu_tracker.update(image);localized=tr['valid'] and cr['valid'];status_parity=status_parity and tr['valid']==cr['valid'];uv=tr['uv'];score=tr.get('score',0.)
                        if localized:parity.append(float(np.linalg.norm(np.asarray(uv)-cr['uv'])))
                    else:
                        uv,score=locate_static(image,loc['bank']);cu,cs=locate_static(image,loc['bank'],'cpu');parity.append(float(np.linalg.norm(uv-cu)));localized=True
                    loc_error=float(np.linalg.norm(plane_point(uv,z,loc['calibration'])[:2]-s['cup_world'][:2])*1000) if localized else None
                    small=np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy();px,py=np.rint(project(np.r_[s['cup_world'][:2],z],cal)*384/1024).astype(int)
                    small[max(0,py-24):min(384,py+48),max(0,px-50):min(384,px+50)]=180
                    _,ns=centers(coarse(torch.tensor(small,device='cuda')[None]));negative_scores.append(float(ns[0]))
                    x=torch.tensor(cup_crop(image,xy,cal),device='cuda')[None];a=primary_prediction(x,xy,s['declared_cup_geometry'],rgb);b,visible=color_prediction(x,xy,s['declared_cup_geometry'],color)
                    ca=primary_prediction(x.cpu(),xy,s['declared_cup_geometry'],rgb);cb,cv=color_prediction(x.cpu(),xy,s['declared_cup_geometry'],color)
                    j=int(np.abs(sample_times-times[i]).argmin());assert abs(sample_times[j]-times[i])<1e-6;target=samples[j]['cup_fill_fraction']
                    valid=bool(visible[0] and cv[0]);av=float(a[0]);bv=float(b[0]) if valid else None
                    frames.append(dict(index=int(i),t=float(times[i]),target=target,localization_valid=localized,localization_error_mm=loc_error,tracker_reason=tr['reason'] if tracking else None,score=score,rgb=av,color=bv,color_valid=valid,level_parity=max(abs(av-float(ca[0])),abs(bv-float(cb[0]))) if valid else None))
        complete=len(frames)==40 and all(v['color_valid'] for v in frames)
        rms=error_metrics([v['rgb'] for v in frames],[v['target'] for v in frames]) if complete else dict(valid=False)
        cms=error_metrics([v['color'] for v in frames],[v['target'] for v in frames]) if complete else dict(valid=False)
        diff=max((abs(v['rgb']-v['color']) for v in frames if v['color_valid']),default=None)
        passed=bool(physical and complete and status_parity and (not tracking or max(initial_raw_errors)<g['localization_max_error_mm']) and anchor_error is not None and anchor_error<g['localization_max_error_mm'] and localization_ok(frames,g['localization_max_error_mm'],loc['threshold']) and level_ok(rms,g) and level_ok(cms,g) and diff<g['crosscheck_max_disagreement'] and max(parity)<.1 and max(v['level_parity'] for v in frames)<g['cpu_gpu_level_difference'] and max(negative_scores)<loc['threshold'] and min(v['score'] for v in frames)-max(negative_scores)>g['localization_negative_margin'])
        row=dict(name=name,passed=passed,physical=physical,source_sha256=r['sha256'],anchor=anchor,anchor_error_mm=anchor_error,initial_raw_errors_mm=initial_raw_errors,tracker_status_parity=status_parity,frames=frames,rgb_metrics=rms,color_metrics=cms,max_disagreement=diff,locator_cpu_gpu_max_pixels=max(parity,default=None),synthetic_absent_max=max(negative_scores,default=None));rows.append(row)
        (ROOT/'moving-holdout-progress.json').write_text(json.dumps(dict(cases=rows,complete=False),indent=2)+'\n');print(json.dumps(dict(name=name,passed=passed,rgb=rms,color=cms)),flush=True)
    assert all(sha(ROOT/n)==v for n,v in selection['frozen_files'].items())
    report=dict(passed=all(r['passed'] for r in rows),passed_cases=sum(r['passed'] for r in rows),cases=rows,selection_sha256=sha(ROOT/'component-selection.json'),audit_sha256=sha(ROOT/'holdout-r0-audit.json'),review_sha256=sha(ROOT/'holdout-r0-manual-review.json'),autonomy_accepted=False,scope='Once-only 18 declared known-size moving perception scenes; every 20 Hz motion frame checked. Authored bottle motion, no autonomous grasp/pour claim. No fitting, retries or threshold changes.')
    with (ROOT/'moving-holdout-evaluation.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=report['passed'],passed_cases=report['passed_cases'])),flush=True)
if __name__=='__main__':main()
