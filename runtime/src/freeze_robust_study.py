"""Bind 30 new cases to the byte-identical previously tested controller."""
import datetime,hashlib,json,os,shutil,urllib.request
from pathlib import Path
from pour_robust_study import ROOT,PARENT,PREFIX,CHECKPOINT_SHA,RUNTIME_SOURCES,plan


def main():
    assert not list(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json')),'Study already started'
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5))
    assert (h.get('ready') or h.get('phase')=='Ready') and not h['queue'] and not h.get('recording',{}).get('active')
    assert not h.get('error') and not h.get('learning_error')
    assert hashlib.sha256((PARENT/'best.pt').read_bytes()).hexdigest()==CHECKPOINT_SHA
    frozen=plan();shutil.copyfile(PARENT/'best.pt',ROOT/'best.pt')
    parent=json.loads((PARENT/'final-selection.json').read_text());sources={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in RUNTIME_SOURCES}
    unchanged={n:sha for n,sha in parent['runtime_sources'].items() if n!='run_fountain.py'}
    assert all(sources[n]==sha for n,sha in unchanged.items()),'Actor/model/physics changed since six-case benchmark'
    r=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=CHECKPOINT_SHA,
        scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),source_commit=os.environ['AIWORKER_SOURCE_COMMIT'],
        runtime_sources=sources,unchanged_policy_physics_sources=unchanged,parent_checkpoint_path=str(PARENT/'best.pt'),
        parent_plan_sha256=parent['scene_plan_sha256'],parent_source_commit=parent['source_commit'],
        scheduling_change='run_fountain adds only the robust_test queue command; model, Actor, scoring, cameras, material and physics unchanged',
        expected_test_names=[s['name'] for s in frozen['test']],test_policy='All30 once; no fitting, threshold changes or tuning during batch; retain every failure',
        primary_definition='Verified sustained grasp, >=90%rim OR source<=max(.25 nominal mL,1%initial), correct reason, complete upright return/release, zero outside/below-worktop particles',
        legacy_definition='Original >=90%nominal inventory criterion retained separately',volume_units='Nominal particle-count volumes, not calibrated physical mL')
    (ROOT/'final-selection.json').write_text(json.dumps(r,indent=2)+'\n')
    shutil.copyfile(PARENT/'model-summary.json',ROOT/'model-summary.json')
    print(json.dumps({k:r[k] for k in ('frozen_utc','selected_checkpoint_sha256','scene_plan_sha256','source_commit')},indent=2))


if __name__=='__main__':main()
