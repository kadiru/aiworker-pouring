"""Initialize calibration from recorded physical cup particles, not compressed grids."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from bottle_pour import CUP,vessel_mask
from pour_finish_calibration import cup_points
from pour_finish_study import ROOT

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
source=Path('/workspace/outputs/fountain/20261102T194804Z/076-robot_full-790609')
spec=json.loads((source/'parameters.json').read_text());metrics=json.loads((source/'metrics.json').read_text())
plan=json.loads((ROOT/'calibration-plan.json').read_text())
assert len(plan['cases'])==51 and plan['cases'][3]['name'].endswith('_dense')
physics=('spacing','viscosity','cohesion','surface_tension','friction','adhesion','damping','vorticity','iterations','neighbors','dt','contact_ratio')
assert all(spec[k]==plan['cases'][3][k] for k in physics)
with np.load(source/'trajectory.npz',allow_pickle=False) as d:
    key=sorted(k for k in d if k.startswith('positions_'))[-1]
    rotation=np.array(metrics['scene']['cup_frame_rotation']);origin=np.array(metrics['scene']['cup_frame_world'])
    local=(d[key]-origin)@rotation
    points=local[vessel_mask(local,CUP)].astype(np.float32)
seed=ROOT/'calibration-particle-seed.npz';assert not seed.exists()
np.savez_compressed(seed,cup_local_points=points)
provenance=dict(source=str(source),trajectory_sha256=digest(source/'trajectory.npz'),parameters_sha256=digest(source/'parameters.json'),
    source_final_key=key,source_cup_count=len(points),source_task_had_spill=True,
    scope='Only particles physically inside the historical cup are reused as a calibration initializer. This is not a new task success; no escaped particles enter the new cup. Initial velocities are zero, followed by native settling.',
    seed_sha256=digest(seed),physics={k:spec[k] for k in physics})
new=plan['cases'][:3]
for i,old in enumerate(plan['cases'][3:]):
    s=dict(old,name=old['name'].removesuffix('_dense')+'_settled',calibration_version=2,
        calibration_initial_height=(.86,.92,.97)[i%3],calibration_particle_seed=str(seed),calibration_particle_seed_sha256=digest(seed))
    s.pop('calibration_pitch_fraction',None);s['cup_initial_ml']=len(cup_points(s))*s['spacing']**3*1e6;new.append(s)
write(ROOT/'calibration-plan-dense-v1.json',plan)
write(ROOT/'calibration-seed-provenance.json',provenance)
write(ROOT/'calibration-plan.json',dict(cases=new,seed=provenance,excluded_dense_pilots=[s['name'] for s in plan['cases'][3:6]],
    scope='Three stable low-level pilots plus 48 stationary settled-seed initializations, 12 training and four validation layouts. No FN30 autonomous development/test images.'))
print(json.dumps(dict(cases=len(new),seed=provenance,first_seed_volumes=[s['cup_initial_ml'] for s in new[3:6]]),indent=2))
