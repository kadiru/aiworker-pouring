"""Independent containment and sleep/height comparison of cold-start fixtures."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/src')
import numpy as np
from bottle_pour import vessel_mask,CUP
from robot_pour import FRAME
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--part',choices=('impact','impact_followup'),default='impact');a=p.parse_args()
plan=read(ROOT/'plan.json');manifest=read(ROOT/'candidate.json');expected={s['name']:s for s in plan[a.part]};destination=ROOT/(a.part+'-outcomes.json')
assert len(expected)==(12 if a.part=='impact' else 2) and not destination.exists()
seed_path=Path(plan[a.part][0]['impact_state_seed']);seed=np.load(seed_path);ids=seed['original_ids'];tracked=[int(np.flatnonzero(ids==i)[0]) for i in (114,1437)]
audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete'];index={r['name']:r for r in audit['records']};rows=[]
for name,spec in expected.items():
    record=index[name];folder=Path(record['path']).parent
    assert sha(folder/'episode.hdf5')==record['sha256']
    for n,d in record['sidecars'].items():assert sha(folder/n)==d
    actual=read(folder/'parameters.json');assert all(actual[k]==v for k,v in spec.items())
    samples=read(folder/'samples.json');metrics=read(folder/'metrics.json');data=np.load(folder/'trajectory.npz');times=data['times']
    cup=np.asarray(spec['cup_world']);final=data[f'positions_{len(times)-1:03}'];inside=vessel_mask((final-cup)@FRAME,CUP)
    tracked_rows=[]
    for source_id,i in zip((114,1437),tracked):
        xyz=np.array([(data[f'positions_{j:03}'][i]-cup)@FRAME for j in range(len(times))]);vel=np.array([data[f'velocities_{j:03}'][i]@FRAME for j in range(len(times))])
        tracked_rows.append(dict(original_id=source_id,seed_index=i,contained_final=bool(inside[i]),ever_near_table=bool(np.any(xyz[:,2]<.0035)),max_sampled_upward_speed_m_s=float(vel[:,2].max()),max_sampled_height_m=float(xyz[:,2].max())))
    sleep=[];previous=None
    for s in samples:
        state=s.get('sleep_diagnostic')
        if state!=previous:sleep.append(dict(t=s['t'],state=state,level=s['cup_fill_fraction']));previous=state
    rows.append(dict(name=name,zero_measured_loss=bool(metrics['zero_spill']),spill=metrics['spill'],tracked_drops=tracked_rows,first_recorded_time_s=float(times[0]),
        first_level=samples[0]['cup_fill_fraction'],final_level=samples[-1]['cup_fill_fraction'],max_level=max(s['cup_fill_fraction'] for s in samples),
        sleep_transitions=sleep,source_sha256=record['sha256'],trajectory_sha256=sha(folder/'trajectory.npz'),
        controls={k:spec.get(k,1. if k in ('max_depenetration_velocity','cfl_coefficient') else None) for k in ('viscosity','cohesion','surface_tension','friction','iterations','max_depenetration_velocity','cfl_coefficient')}))
report=dict(part=a.part,cases=rows,baseline_reproduces_loss=not rows[0]['zero_measured_loss'],contained_variants=[r['name'] for r in rows if r['zero_measured_loss']],
    scope='Non-autonomous cold-start fluid impact. All recorded particles outside the bottle, including both tracked drops; remaining source liquid and moving bottle are omitted. Camera initialization advances 0.05 seconds before transition recording; first_recorded_time_s reports the first saved particle sample. Differences support only this controlled fixture. A zero-loss baseline cannot establish a repair. No force/impulse or continuous peak measurement.',
    audit_sha256=sha(ROOT/'recording-audit.json'),seed_sha256=sha(seed_path),helper_sha256=sha(Path(__file__)))
destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(part=a.part,cases=[dict(name=r['name'],zero_loss=r['zero_measured_loss'],first_level=r['first_level'],final_level=r['final_level']) for r in rows])))
