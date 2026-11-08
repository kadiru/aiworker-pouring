"""Independently recompute diagnostics from saved native particle trajectories."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from bottle_pour import vessel_mask
from pour_vessel_geometry import cup_profile
from pour_perception_geometry import upper_surface,settled_label
from robot_pour import FRAME
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[]
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*PV_A_geometry_*/trajectory.npz')):
        spec=json.loads((path.parent/'parameters.json').read_text());rows=json.loads((path.parent/'samples.json').read_text())
        p=cup_profile(spec);errors=[];surface=[]
        with np.load(path) as f:
            for i,t in enumerate(f['times']):
                r=min(rows,key=lambda r:abs(r['t']-t));assert abs(r['t']-t)<.0001
                local=(f[f'positions_{i:03}']-spec['cup_world'])@FRAME
                # The stationary bottle envelope is wholly above/away from the
                # cup interior in these fixtures; all contained particles match.
                mask=vessel_mask(local,p);assert abs(mask.sum()*spec['spacing']**3*1e6-r['cup_ml'])<1e-7
                value=upper_surface(local[mask],p,spec['spacing'])
                errors.append(abs(value['height_fraction']-r['cup_column_height_fraction']))
                surface.append(dict(t=float(t),column=value['height_fraction'],legacy=r['cup_fill_fraction']))
        records.append(dict(name=spec['name'],trajectory_sha256=sha(path),samples_sha256=sha(path.parent/'samples.json'),
            snapshots=len(errors),recompute_max_error=max(errors),inventory_nominal_ml=rows[-1]['cup_ml'],
            legacy_label=settled_label(rows),column_label=settled_label(rows,field='cup_column_height_fraction'),snapshots_measured=surface))
    assert len(records)==3
    pair=records[:2];old_diff=pair[1]['legacy_label']['target']-pair[0]['legacy_label']['target']
    new_diff=pair[1]['column_label']['target']-pair[0]['column_label']['target']
    passed=all(r['recompute_max_error']<1e-8 and r['legacy_label']['accepted'] and r['column_label']['accepted']
        and abs(r['legacy_label']['target']-r['column_label']['target'])<.01 for r in records) and abs(new_diff-old_diff)<.005
    report=dict(passed=passed,records=records,equal_inventory_pair=True,legacy_pair_difference=old_diff,column_pair_difference=new_diff,
        selected_training_target='Per-record final-second median of the original measured surface; no constant inventory-to-height lookup.',
        interpretation='Independent spatial top measurements retain the paired surface-height difference. Equal particle inventory does not establish equal settled height in this native PBD fixture; the exact physical/numerical cause is not determined here.',
        unchanged='Original instantaneous task scores and particle/liquid/collision parameters.',source_sha256=sha(Path(__file__)))
    (ROOT/'surface-label-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
