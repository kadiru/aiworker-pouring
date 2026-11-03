import json
from pathlib import Path
import numpy as np
import h5py
root=Path('/workspace')
probe=json.loads((root/'outputs/quest-ik-probe.json').read_text())
state=json.loads((root/'outputs/quest-status.json').read_text())
names=state['dof_names']; arm=[i for i,n in enumerate(names) if n.startswith('arm_r_')]
grip=[i for i,n in enumerate(names) if n.startswith('gripper_r_')]
q=np.array(probe['samples']);initial=np.array(probe['initial_q'])
resets=json.loads((root/'outputs/quest-dry-resets.json').read_text())
report={'input_source':'synthetic controller poses through recovered retargeter and DLS IK','frames':len(q),
        'finite':bool(np.isfinite(q).all()),'arm_peak_delta_rad':float(np.abs(q[:,arm]-initial[arm]).max()),
        'gripper_peak_delta_rad':float(np.abs(q[:,grip]-initial[grip]).max()),
        'arm_return_error_rad':float(np.abs(q[-1,arm]-initial[arm]).max()),
        'three_resets':len(resets)==3 and all(r['ready'] and not r['recording'] and r['samples']==0 for r in resets),
        'reset_position_spread_m':float(max(np.ptp(np.array([r[k][:3] for r in resets]),axis=0).max() for k in ('source_cup_pose','target_cup_pose')))}
file=root/'datasets/quest_pouring_20261028_212741.hdf5'
with h5py.File(file,'r') as f:
    group=f['data/demo_0']
    report['hdf5']={'file':str(file),'samples':int(group.attrs['num_samples']),'actions_shape':list(group['actions'].shape),'joint_states_shape':list(group['states/articulation/robot/joint_position'].shape)}
report['passed']=report['finite'] and report['arm_peak_delta_rad']>.01 and report['gripper_peak_delta_rad']>.5 and report['arm_return_error_rad']<.005 and report['three_resets'] and report['reset_position_spread_m']<1e-4 and report['hdf5']['samples']==160
(root/'outputs/quest-dry-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report['passed']
