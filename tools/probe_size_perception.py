"""Frozen-weight CPU probe at low fill; no fitting or privileged crop positions."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_models import CupEstimator
from pour_guard_vision import locate,plane_point,cup_crop,surface_features
from pour_settled_model import features,predict
from pour_size_study import ROOT,OUTPUTS
from pour_temporal_localization import consensus


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    model=CupEstimator().eval();model.load_state_dict(saved['cup_model'])
    plan=json.loads((ROOT/'plan.json').read_text());expected={s['name']:s for s in plan['calibration_seed']}
    records=[]
    for path in sorted(OUTPUTS.glob('*/*SZ_A_cup_seed_*/episode.hdf5')):
        spec=json.loads((path.parent/'parameters.json').read_text());assert spec['name'] in expected
        samples=json.loads((path.parent/'samples.json').read_text())
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            final=decode(d['next_obs/task_image'][-1],'jpeg95');history=[];accepted=None
            config=saved['localization_consensus'];times=d['time'][:]
            for i,t in enumerate(times):
                if t-times[0]>config['timeout_s']:break
                initial=decode(d['obs/task_image'][i],'jpeg95')
                uv,score=locate(initial,saved['cup_bank'],device='cpu')
                if i==0:first_uv=uv.copy();first_score=score
                history.append((uv.tolist(),score))
                accepted=consensus(history,saved['thresholds']['cup_localization_score'],config['cup'])
                if accepted is not None:break
        cal=copy.deepcopy(saved['calibration']);cal['cup_geometry']=spec['declared_cup_geometry']
        first_xy=plane_point(first_uv,cal['table_z']+cal['cup_geometry']['rim_z_m'],cal)[:2]
        xy=cnn=geo=estimate=error=None
        if accepted:
            xy=plane_point(accepted['uv'],cal['table_z']+cal['cup_geometry']['rim_z_m'],cal)[:2]
            image=torch.from_numpy(cup_crop(final,xy,cal))[None]
            cnn=float((model(image)[0]+saved['cup_bias']).clamp(0,1))
            f,_,n=surface_features(image);g=saved['surface_geometry']
            geo=float((f[0,:g['dimensions']]@torch.as_tensor(g['coef'])+g['bias']).clamp(0,1)) if n[0]>1 else 0.
            estimate=float(predict(features(image,torch.tensor([cnn]),xy),saved['settled_regressor'])[0])
            error=float(np.linalg.norm(xy-np.asarray(spec['cup_world'][:2]))*1000)
        actual=samples[-1]['cup_fill_fraction']
        records.append(dict(name=spec['name'],height_scale=spec['cup_height_scale'],diameter_scale=spec['cup_diameter_scale'],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first_cup_localization_score=first_score,
            first_frame_localization_error_mm=float(np.linalg.norm(first_xy-np.asarray(spec['cup_world'][:2]))*1000),
            temporal_confirmation=accepted,frames_considered=len(history),
            localized_xy=None if xy is None else xy.tolist(),localization_error_mm=error,
            measured_final_height=actual,cup_cnn=cnn,surface_geometry=geo,settled_prediction=estimate,
            settled_error=None if estimate is None else estimate-actual))
    assert len(records)==9
    confirmed=[r for r in records if r['temporal_confirmation'] is not None]
    report=dict(scope='Nine new low-fill stationary size pilots, all at one consumed layout. Frozen model with its actual cup temporal-confirmation rule, known declared cup dimensions and image-localized XY; simulator fixture position/height are scoring labels only. Bottle localization is not assessed. No fitting, no near-full guarantee, no autonomous task success or unknown-size inference claim.',
        checkpoint_sha256=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest(),device='cpu',
        cases=records,confirmed_cups=len(confirmed),unconfirmed_cups=9-len(confirmed),
        max_localization_error_mm=max((r['localization_error_mm'] for r in confirmed),default=None),
        settled_mae=None if not confirmed else float(np.mean([abs(r['settled_error']) for r in confirmed])),
        settled_max_abs=max((abs(r['settled_error']) for r in confirmed),default=None),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'low-fill-temporal-perception-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','scope')}))


if __name__=='__main__':main()
