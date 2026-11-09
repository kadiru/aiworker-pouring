"""Independent support-slide scoring, with no policy or candidate mutation."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/src')
import numpy as np
from robot_pour import FRAME
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
destination=ROOT/'support-outcomes.json';assert not destination.exists()
plan=read(ROOT/'plan.json');audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete']
index={r['name']:r for r in audit['records']};rows=[]
for spec in plan['support']:
    record=index[spec['name']];path=Path(record['path']);assert sha(path)==record['sha256']
    for n,d in record['sidecars'].items():assert sha(path.parent/n)==d
    actual=read(path.parent/'parameters.json');assert all(actual[k]==v for k,v in spec.items())
    samples=read(path.parent/'samples.json');metrics=read(path.parent/'metrics.json')
    initial=(np.asarray(spec['cup_world'])+FRAME@np.r_[spec['source_xy'],.0002])[:2]
    final=np.asarray(samples[-1]['bottle_pose_xyz'][:2]);previous=np.asarray(samples[-6]['bottle_pose_xyz'][:2]);dt=samples[-1]['t']-samples[-6]['t']
    other_contacts=sorted({c for s in samples for c in s['contacts'] if c!='Bench'})
    rows.append(dict(name=spec['name'],initial_xy=initial.tolist(),final_xy=final.tolist(),drift_m=float(np.linalg.norm(final-initial)),
        final_speed_m_s=float(np.linalg.norm(final-previous)/dt),other_contacts=other_contacts,zero_spill=metrics['zero_spill'],
        authored_support_contact=metrics['scene'].get('bottle_support_contact'),source_sha256=record['sha256']))
assert len(rows)==2
passed=rows[0]['drift_m']>=.015 and rows[1]['drift_m']<=.005 and rows[1]['final_speed_m_s']<=.002 and all(r['zero_spill'] and not r['other_contacts'] for r in rows)
report=dict(cases=rows,passed=passed,audit_sha256=sha(ROOT/'recording-audit.json'),helper_sha256=sha(Path(__file__)),
    scope='Two-second authored-velocity rigid support comparison; no autonomous grasp/return success claim and no calibrated real material coefficient. Unchanged strict/side/headroom criteria still require six new accepted development pours.')
destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
