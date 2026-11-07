"""Read-only CPU localization/level probe on closed calibration recordings."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path[:0]=['/workspace/learning/A-finish-20261106-v1/staged','/workspace/src','/workspace']
from pour_finish_study import ROOT,PARENT,OUTPUTS,sha
from pour_guard_vision import locate,plane_point,cup_crop
from pour_guard_models import CupEstimator
from pour_settled_model import features,predict
from validate_pour_recording import h5py,decode

torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--reference-anchors',action='store_true');args=p.parse_args()
saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
model=CupEstimator().eval();model.load_state_dict(saved['cup_model'])
expected={s['name']:s for s in json.loads((ROOT/'calibration-plan.json').read_text())['cases']};rows=[]
anchors=None
if args.reference_anchors:
    from pour_finish_train import reference_crop_anchors
    anchors=reference_crop_anchors(saved,list(expected.values()),device='cpu')
with torch.inference_mode():
    for path in sorted(OUTPUTS.glob('*/*FCAL_A_*/episode.hdf5')):
        spec=json.loads((path.parent/'parameters.json').read_text())
        if spec['name'] not in expected:continue
        assert spec==expected[spec['name']]
        samples=json.loads((path.parent/'samples.json').read_text())
        with h5py.File(path) as f:
            d=f['data/demo_0']
            if anchors is None:
                uv=np.median([locate(decode(d['obs/task_image'][i],'jpeg95').copy(),saved['cup_bank'],device='cpu')[0] for i in range(8)],axis=0)
                xy=plane_point(uv,1.078,saved['calibration'])[:2]
            else:xy=np.array(anchors[tuple(spec['cup_world'])+tuple(spec['source_xy'])]['cup_xy'])
            crop=cup_crop(decode(d['next_obs/task_image'][-1],'jpeg95'),xy,saved['calibration']);a=torch.tensor(crop)[None]
            cnn=(model(a)+saved['cup_bias']).clamp(0,1);estimated=float(predict(features(a,cnn,xy),saved['settled_regressor'])[0])
        rows.append(dict(name=spec['name'],source_sha256=sha(path),split=spec['study_split'],cup_xy=xy.tolist(),
            localization_error_mm=float(np.linalg.norm(xy-np.array(spec['cup_world'][:2]))*1000),
            true_level=samples[-1]['cup_fill_fraction'],parent_predicted_level=estimated,parent_error=estimated-samples[-1]['cup_fill_fraction']))
assert rows and not torch.cuda.is_initialized()
report=dict(records=rows,reference_anchors=list(anchors.values()) if anchors else None,closed=len(rows),device='cpu',cuda_initialized=False,parent_checkpoint_sha256=sha(PARENT/'best.pt'),
    maximum_localization_error_mm=max(r['localization_error_mm'] for r in rows),
    parent_level_mae=float(np.mean([abs(r['parent_error']) for r in rows])),
    scope='Calibration diagnostic only; no fitting, checkpoint mutation or autonomous outcome claim. Simulator poses are labels for this probe, never runtime controller input.')
(ROOT/('calibration-reference-anchor-probe.json' if anchors else 'calibration-localization-probe.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('records','reference_anchors')},indent=2))
