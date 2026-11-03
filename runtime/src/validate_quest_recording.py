import json
from pathlib import Path
import numpy as np
import h5py
root=Path('/workspace')
with h5py.File(root/'outputs/quest-last-export.hdf5','r') as f:
 g=f['data/demo_0']; count=int(g.attrs['num_samples'])
 report={'samples':count,'attributes':{k:v.item() if hasattr(v,'item') else v for k,v in g.attrs.items()},'arrays':{},'cameras':{}}
 for key in ('actions','fluid_positions','timestamps_unix','states/articulation/robot/joint_position','obs/eef_pos'):
  a=g[key][:]; report['arrays'][key]={'shape':list(a.shape),'finite':bool(np.isfinite(a).all())}
 for key in ('head_cam','right_wrist_cam','overhead_cam'):
  a=g['obs/'+key][:]; report['cameras'][key]={'shape':list(a.shape),'dtype':str(a.dtype),'std':float(a.std()),'temporal_change_mean':float(np.abs(a[-1].astype(float)-a[0].astype(float)).mean())}
 report['particle_peak_motion_m']=float(np.linalg.norm(g['fluid_positions'][:]-g['fluid_positions'][0],axis=2).max())
 report['wall_seconds']=float(g['timestamps_unix'][-1,0]-g['timestamps_unix'][0,0])
 report['mean_action_hz']=(count-1)/report['wall_seconds']
 report['passed']=count==160 and report['attributes']['input_source']=='synthetic_validation' and not report['attributes']['outcome_verified'] and all(v['finite'] and v['shape'][0]==count for v in report['arrays'].values()) and all(v['std']>5 and v['shape']==[count,244,244,3] for v in report['cameras'].values()) and report['particle_peak_motion_m']>.005
(root/'outputs/quest-recording-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report['passed']
