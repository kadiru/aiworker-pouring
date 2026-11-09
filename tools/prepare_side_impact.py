"""Prepare a recorded-state fluid-impact experiment; never activates a policy."""
import copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/src')
import numpy as np
from bottle_pour import BOTTLE,CUP,vessel_mask
from scipy.spatial.transform import Rotation
from robot_pour import FRAME
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'impact-seed-v2.npz').exists()
paths=list(Path('/workspace/outputs/fountain').glob('*/*SR_A_diagnostic_00_04-*/metrics.json'));assert len(paths)==1
folder=paths[0].parent;spec=read(folder/'parameters.json');data=np.load(folder/'trajectory.npz')
i=int(np.argmin(abs(data['times']-24.005)));assert abs(data['times'][i]-24.005)<1e-5
points=data[f'positions_{i:03}'];velocities=data[f'velocities_{i:03}'];cup=np.array(spec['cup_world'])
sample=min(read(folder/'samples.json'),key=lambda s:abs(s['t']-float(data['times'][i])))
assert abs(sample['t']-float(data['times'][i]))<1e-6
rotation=Rotation.from_quat(sample['bottle_quat_xyzw']).as_matrix()
keep=~vessel_mask((points-np.array(sample['bottle_pose_xyz']))@rotation,BOTTLE)
assert keep[114] and keep[1437]
ids=np.flatnonzero(keep)
np.savez_compressed(ROOT/'impact-seed-v2.npz',positions=points[keep],velocities=velocities[keep],original_ids=ids)
seed_sha=sha(ROOT/'impact-seed-v2.npz')
variants=[('baseline',{}),('water_viscosity',dict(viscosity=.0000017)),
 ('water_cohesion',dict(cohesion=.01)),('water_surface',dict(surface_tension=.0074)),
 ('water_preset',dict(cohesion=.01,friction=.1,surface_tension=.0074,viscosity=.0000017)),
 ('lower_depenetration',dict(max_depenetration_velocity=.25)),('iterations32',dict(iterations=32)),
 ('water_iterations32',dict(cohesion=.01,friction=.1,surface_tension=.0074,viscosity=.0000017,iterations=32)),('cfl_half',dict(cfl_coefficient=.5)),('cfl_quarter',dict(cfl_coefficient=.25)),('cfl_half_iterations32',dict(cfl_coefficient=.5,iterations=32)),('cfl_quarter_iterations32',dict(cfl_coefficient=.25,iterations=32))]
rows=[]
for index,(label,changes) in enumerate(variants):
 s=copy.deepcopy(spec);s.update(name=f'SR_A_impact_{index:02}_{label}',study_split='impact',calibration_only=True,policy_evaluation=False,
  initial_ml=spec['spacing']**3*1e6,cup_initial_ml=len(ids)*spec['spacing']**3*1e6,max_seconds=2.,
  impact_state_seed=str(ROOT/'impact-seed-v2.npz'),impact_state_seed_sha256=seed_sha,trajectory_stride=1,trajectory_start_s=0.,record_sleep_diagnostics=True,
  evidence_scope='Non-autonomous cold-start fluid impact: all recorded particles outside the bottle, including the two observed escaped droplets; no bottle motion or policy. Baseline reproduction is required before interpreting variants.',**changes)
 rows.append(s)
report=dict(impact=rows,seed_sha256=seed_sha,source_trajectory_sha256=sha(folder/'trajectory.npz'),source_parameters_sha256=sha(folder/'parameters.json'),
 source_time=float(data['times'][i]),particle_count=len(ids),original_ids_stored_in_seed=True,selected_drop_ids=[114,1437],
 water_preset_source='/isaac-sim/extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physx/scripts/particleUtils.py:AddPBDMaterialWater',stage_meters_per_unit=1.,
 scope='Preparation only. No simulator state is changed and no policy uses privileged state. Original cup/bottle geometry and native source recording are preserved.')
(ROOT/'impact-plan-v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(variants=len(rows),particles=len(ids),seed_sha256=seed_sha)))
