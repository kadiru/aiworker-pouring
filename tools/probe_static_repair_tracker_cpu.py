"""Early CPU diagnostic on closed training records; no fitting/admission claim."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from dynamic_cup_tracker import CupTracker
from validate_pour_recording import h5py,decode
from pour_guard_vision import plane_point
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def main():
    torch.set_num_threads(2)
    assert not (ROOT/'static-repair-early-tracker-cpu.json').exists()
    progress=json.loads((ROOT/'train-r2/progress.json').read_text());rows=[]
    assert progress['cases'];loc=torch.load(ROOT/'tracker-t1.pt',map_location='cpu',weights_only=False)
    for c in progress['cases'][:2]:
        directory=Path(c['path']);s=json.loads((directory/'parameters.json').read_text());tracker=CupTracker(loc,s['declared_cup_geometry'],'cpu')
        with h5py.File(directory/'episode.hdf5') as f:
            d=f['data/demo_0'];result=None
            for i in range(12):
                result=tracker.update(decode(d['obs/cup_detail_image'][i],'jpeg95'))
                if result['valid']:break
            err=None;frames=[]
            if result['valid']:
                xy=plane_point(result['uv'],1.04+s['declared_cup_geometry']['rim_z_m'],loc['calibration'])[:2]
                err=float(np.linalg.norm(xy-s['cup_world'][:2])*1000)
                times=d['next_time'][:]
                for i in np.flatnonzero(times>=times[-1]-1.-1e-7)[::2]:
                    r=tracker.update(decode(d['next_obs/cup_detail_image'][int(i)],'jpeg95'));frames.append(dict(index=int(i),valid=r['valid'],reason=r['reason']))
            rows.append(dict(name=s['name'],confirmed=result['valid'],error_mm=err,frames=frames,necessary_check=err is not None and err<3 and len(frames)==11 and all(r['valid'] for r in frames)))
    report=dict(cases=rows,scope='Early CPU diagnostic on at most two closed native training records; complete native audit, actual review and full GPU admission remain required.')
    (ROOT/'static-repair-early-tracker-cpu.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
