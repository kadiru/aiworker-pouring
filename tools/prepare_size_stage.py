"""Inside the owned container: prepare a separate, explicitly untrained candidate."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path('/workspace/learning/A-size-20261108-v1')
PARENT=Path('/workspace/learning/A-finish-20261106-v1')
PARENT_SHA='5068af6b18583afcb3eff231110c63896e0be898f389b9aad940f9e9405179d3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')


def main():
    sys.path.insert(0,str(ROOT/'staged'))
    from pour_source_watchdog import DEFAULT
    from pour_vessel_geometry import cup_geometry,initial_cup_points
    assert not (ROOT/'candidate.json').exists()
    assert sha(PARENT/'best.pt')==PARENT_SHA
    assert (PARENT/'final-selection.json').exists()
    original=read(PARENT/'model-summary.json')['runtime_sources']
    for n,h in original.items():assert sha(Path('/workspace/src')/n)==h,n
    saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
    saved=copy.deepcopy(saved)
    saved['source_watchdog']=DEFAULT
    saved['model_version']='visual_size_stage_v25_recovery_only'
    saved['parent_checkpoint_sha256']=PARENT_SHA
    saved['method']='Unchanged V24 learned weights with a programmed bounded source-uncertainty recovery supervisor; no new fit.'
    sources={p.name:sha(p) for p in sorted((ROOT/'staged').glob('*.py')) if not p.name.startswith('test_')}
    saved['sources']=sources
    torch.save(saved,ROOT/'best.pt')
    parent_plan=read(PARENT/'frozen-plan.json')
    # Diagnosed old test layouts become regression development, never fresh tests.
    indices=(13,23,15,19,0,29)
    recovery=[]
    for i in indices:
        spec=copy.deepcopy(parent_plan['test'][i])
        spec.update(name=f'SZ_A_recovery_{i:02d}',study_split='development',
            former_test_case=spec['name'],checkpoint=str(ROOT/'best.pt'),development_round=0)
        recovery.append(spec)
    base=copy.deepcopy(parent_plan['development'][0])
    calibration=[]
    for i,(h,d) in enumerate((h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1)):
        spec=dict(base,name=f'SZ_A_cup_static_{i:02d}',study_split='calibration_pilot',
            policy_evaluation=False,calibration_only=True,max_seconds=2.5,
            cup_height_scale=h,cup_diameter_scale=d,cup_initial_ml=16*h*d*d,
            initial_ml=20.,calibration_scope='Stationary containment/camera pilot, not pouring or a learned size test')
        spec.pop('checkpoint',None)
        spec['declared_cup_geometry']=cup_geometry(spec)
        assert len(initial_cup_points(spec))>100
        calibration.append(spec)
    write('plan.json',dict(version=1,parent_checkpoint_sha256=PARENT_SHA,
        calibration=calibration,recovery_development=recovery,
        scope='Six consumed-layout development regressions and nine stationary cup-size pilots. No reserved test results. No bottle scaling or size-aware autonomous claim.'))
    protected={p.name:sha(p) for p in PARENT.iterdir() if p.is_file() and p.suffix in ('.json','.pt')}
    write('candidate.json',dict(checkpoint_sha256=sha(ROOT/'best.pt'),parent_checkpoint_sha256=PARENT_SHA,
        plan_sha256=sha(ROOT/'plan.json'),runtime_sources=sources,source_watchdog=DEFAULT,
        protected_parent_files=protected,trained=False,
        scope=saved['method'],full_prediction_repair_complete=False,
        geometry_policy='Known fixture dimensions for stationary calibration only; camera size inference and resized autonomous paths remain unvalidated.'))
    print(json.dumps(dict(checkpoint_sha256=sha(ROOT/'best.pt'),calibration_cases=9,recovery_cases=6,trained=False)))


if __name__=='__main__':main()
