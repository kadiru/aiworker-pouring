"""Separate follow-up cohort; the completed wide benchmark stays immutable."""
import hashlib,json
from pathlib import Path
ROOT=Path('/workspace/learning/A-finish-20261106-v1')
PARENT=Path('/workspace/learning/A-wide-20261105-v1')
PARENT_SHA='4c8ab7bab4b5121004e47ac1458f6df8ead0bd481c381efa807c7be8195d3430'
PREFIX='FN30_A_'
OUTPUTS=Path('/workspace/outputs/fountain')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plan():return json.loads((ROOT/'frozen-plan.json').read_text())
def seen():return [json.loads(p.read_text()) for p in OUTPUTS.glob(f'*/*{PREFIX}*/parameters.json')]
def development():
    assert not (ROOT/'final-selection.json').exists()
    r=max((s.get('development_round',-1) for s in seen()),default=-1)+1
    return [dict(s,name=s['name']+f'_r{r}',development_round=r) for s in plan()['development']]
def tests():
    frozen=json.loads((ROOT/'final-selection.json').read_text())
    assert sha(ROOT/'best.pt')==frozen['selected_checkpoint_sha256']
    assert sha(ROOT/'frozen-plan.json')==frozen['scene_plan_sha256']
    for n,h in frozen['runtime_sources'].items():assert sha(Path('/workspace/src')/n)==h,n
    started={s['name'] for s in seen()}
    return [s for s in plan()['test'] if s['name'] not in started]

def calibrations():
    assert not (ROOT/'final-selection.json').exists()
    started={json.loads(p.read_text())['name'] for p in OUTPUTS.glob('*/*FCAL_A_*/parameters.json')}
    return [s for s in json.loads((ROOT/'calibration-plan.json').read_text())['cases'] if s['name'] not in started]
