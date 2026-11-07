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
    recovery=ROOT/'development-resume-plan.json'
    if recovery.exists():
        request=json.loads(recovery.read_text())
        assert request['kind']=='recording_infrastructure_resume'
        assert sha(ROOT/'best.pt')==request['checkpoint_sha256']
        assert sha(ROOT/'frozen-plan.json')==request['scene_plan_sha256']
        started={s['name'] for s in seen()}
        original={s['name']:s for s in plan()['development']}
        assert request['round']==0
        assert len({item['name'] for item in request['cases']})==len(request['cases'])
        aborts=json.loads((ROOT/'infrastructure-aborts.json').read_text())['attempts']
        result=[]
        for item in request['cases']:
            base=original[item['planned_name']]
            ordinary=base['name']+f"_r{request['round']}"
            if item.get('retry_of'):
                assert item['retry_of']==ordinary and ordinary in aborts
                assert item['name']==base['name']+f"_retry1_r{request['round']}"
            else:assert item['name']==ordinary
            actual=dict(base,name=item['name'],development_round=request['round'])
            if item.get('retry_of'):actual['infrastructure_retry_of']=item['retry_of']
            if actual['name'] not in started:result.append(actual)
        return result
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
