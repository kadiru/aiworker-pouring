"""Frozen-weight CPU diagnostic on new calibration views; no fitting or admission."""
import argparse
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
from pour_settled_model import features,predict
from pour_temporal_localization import consensus
ROOT=Path('/workspace/learning/A-scale-20261109-v1')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--round',choices=('r1','r2'),required=True);a=p.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    model=CupEstimator().eval();model.load_state_dict(saved['cup_model'])
    audit=read(ROOT/f'pilot-{a.round}-audit.json');assert audit['complete']
    rows=[]
    for r in audit['records']:
        path=Path(r['path']);assert sha(path)==r['sha256'];spec=read(path.parent/'parameters.json')
        cal=copy.deepcopy(saved['cup_localization_calibration']);level_cal=copy.deepcopy(saved['cup_detail_calibration'])
        for c in (cal,level_cal):c['cup_geometry']=spec['declared_cup_geometry']
        with h5py.File(path) as f:
            d=f['data/demo_0'];native_K=f['calibration/cup_detail/intrinsics'][:]
            scaled_K=native_K.copy();scaled_K[:2]*=384/1024
            assert np.allclose(scaled_K,cal['K']) and np.allclose(native_K,level_cal['K'])
            assert np.allclose(d['obs/cup_detail_world_T_camera'][0],cal['T'])
            assert np.allclose(d['obs/cup_detail_world_T_camera'][0],level_cal['T'])
            config=saved['localization_consensus'];history=[];accepted=None
            for i,t in enumerate(d['time'][:]):
                if t-d['time'][0]>config['timeout_s']:break
                image=decode(d['obs/cup_detail_image'][i],'jpeg95')
                image=np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy()
                uv,score=locate(image,saved['cup_bank'],device='cpu');history.append((uv.tolist(),score))
                accepted=consensus(history,saved['thresholds']['cup_localization_score'],config['cup'])
                if accepted is not None:break
            row=dict(name=spec['name'],height_scale=spec['cup_height_scale'],diameter_scale=spec['cup_diameter_scale'],
                     source_sha256=r['sha256'],confirmation=accepted,frames_considered=len(history),
                     stable_label=r['label'],localization_error_mm=None,settled_prediction=None)
            if accepted is not None:
                xy=plane_point(accepted['uv'],cal['table_z']+cal['cup_geometry']['rim_z_m'],cal)[:2]
                row['localization_error_mm']=float(np.linalg.norm(xy-np.asarray(spec['cup_world'][:2]))*1000)
                values=[]
                for index in (-21,-11,-1):
                    image=decode(d['next_obs/cup_detail_image'][index],'jpeg95')
                    x=torch.from_numpy(cup_crop(image,xy,level_cal))[None]
                    cnn=float(np.clip(float(model(x)[0])+saved['cup_bias'],0,1))
                    value=float(predict(features(x,torch.tensor([cnn],dtype=torch.float32),xy),saved['settled_regressor'])[0])
                    geo=float(color_geometry_readout(x,saved['surface_geometry'],xy)[0])
                    values.append(dict(index=index,cnn=cnn,settled=value,color=geo))
                row.update(level_samples=values,settled_prediction=float(np.median([v['settled'] for v in values])),
                           maximum_sample_level_error=max(abs(v['settled']-r['label']['target']) for v in values))
        rows.append(row)
    confirmed=[r for r in rows if r['confirmation'] is not None]
    report=dict(cases=rows,checkpoint_sha256=sha(ROOT/'best.pt'),device='cpu',trained=False,
                confirmed=len(confirmed),max_localization_error_mm=max((r['localization_error_mm'] for r in confirmed),default=None),
                level_mae=None if not confirmed else float(np.mean([abs(r['settled_prediction']-r['stable_label']['target']) for r in confirmed])),
                helper_sha256=sha(Path(__file__)),audit_sha256=sha(ROOT/f'pilot-{a.round}-audit.json'),
                scope='Frozen parent visual models, with explicitly declared cup geometry retargeted in this offline diagnostic only. Image-localized XY, no truth-based crop. One parked layout, static near-full views; no learning, bottle tracking, autonomous control or unknown-size inference. Runtime size gates remain unchanged.')
    with (ROOT/f'pilot-{a.round}-perception-probe.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','scope')}))


if __name__=='__main__':main()
