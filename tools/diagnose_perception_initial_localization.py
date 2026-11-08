"""Replay closed autonomous start frames on CPU; never changes live control."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import locate,plane_point
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    assert a.name.startswith('PV_A_development_') and '/' not in a.name
    paths=list(Path('/workspace/outputs/fountain').glob('*/*'+a.name+'-*/episode.hdf5'));assert len(paths)==1
    path=paths[0];spec=read(path.parent/'parameters.json');metrics=read(path.parent/'metrics.json');decisions=read(path.parent/'policy-decisions.json')
    assert spec['name']==a.name and metrics['policy']['checkpoint_sha256']==sha(ROOT/'best.pt')
    output=ROOT/(a.name+'-localization-diagnostic.json');assert not output.exists()
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);torch.set_num_threads(2);rows=[]
    with h5py.File(path) as f:
        assert f.attrs['status']=='complete';d=f['data/demo_0']
        detail=saved['cup_detail_calibration'];cal=saved['cup_localization_calibration']
        k_error=float(np.max(abs(f['calibration/cup_detail/intrinsics'][:]-np.array(detail['K']))))
        pose_error=float(np.max(abs(d['obs/cup_detail_world_T_camera'][:]-np.array(detail['T']))))
        for i in (0,7,20,39):
            if i>=len(decisions):continue
            rgb=np.asarray(Image.fromarray(decode(d['obs/cup_detail_image'][i],'jpeg95')).resize((384,384),Image.Resampling.BILINEAR)).copy()
            uv,score=locate(rgb,saved['cup_bank'],device='cpu');xy=plane_point(uv,detail['table_z']+.038,cal)[:2]
            rows.append(dict(index=i,uv=uv.tolist(),cpu_score=score,runtime_score=decisions[i]['cup_localization_score'],
                error_mm=float(np.linalg.norm(xy-np.array(spec['cup_world'][:2]))*1000),
                source_temporal_confirmed=decisions[i]['localization_consensus']['source'] is not None,
                cup_temporal_confirmed=decisions[i]['localization_consensus']['cup'] is not None))
    report=dict(name=a.name,source_sha256=sha(path),decisions_sha256=sha(path.parent/'policy-decisions.json'),checkpoint_sha256=sha(ROOT/'best.pt'),
        threshold=saved['thresholds']['cup_localization_score'],intrinsics_max_abs_difference=k_error,pose_max_abs_difference=pose_error,
        observations=rows,helper_sha256=sha(Path(__file__)),scope='Closed native autonomous start frames replayed on CPU with the unchanged checkpoint. Known cup position is an offline error label only. No fitting, altered confidence thresholds, test retries, or live simulator state in control.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
