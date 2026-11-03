"""Freeze model/config/source identity before exactly six new native tests."""
import datetime,hashlib,json,os,urllib.request
from pathlib import Path
from pour_guard_study import ROOT


def main():
    base=Path('/workspace/outputs/fountain');assert not list(base.glob('*/*G_A_test*/parameters.json')),'Tests already started'
    sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest();results=json.loads((ROOT/'guard-results.json').read_text())
    assert results['checkpoint_sha256']==sha
    development=results['groups']['regression']['rows']+results['groups']['validation']['rows']
    assert len(development)==4,'Complete both regressions and both new validation cases first'
    assert all(x['surface_goal_zero_spill_success'] for x in development),'Resolve development task failures before selection'
    health=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5))
    assert health.get('ready') and not health['queue'] and not health.get('recording',{}).get('active'),'Wait for idle/closed recordings'
    assert not health.get('error') and not health.get('learning_error')
    names=('pour_guard_actor.py','pour_guard_train.py','pour_guard_models.py','pour_guard_vision.py',
        'pour_guard_trajectory.py','pour_guard_study.py','pour_guard_data.py','pour_stop_actor.py',
        'pour_stop_models.py','pour_stop_sequence.py','run_fountain.py','robot_pour.py','pour_recording.py',
        'bottle_pour.py','pour_fluid.py','pour_geometry.py','vessel_appearance.py')
    r=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=sha,
        source_commit=os.environ.get('AIWORKER_SOURCE_COMMIT'),
        runtime_sources={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in names},
        scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        selection_reason='New visual/proprioceptive grasp gate and settled-level fill feedback, selected using development only; all prior failed/aborted development retained.',
        development=development,all_development_attempts=results['all_attempts'],
        test_policy='Run six new seed20261122 cases once; no model/config/controller/physics changes, no expert fallback during the block.',
        deadline_note='40 simulated seconds and 12-second grasp deadline allow one retry and settled top-up cycles; success criteria and accepted physics unchanged.',
        expected_test_names=[s['name'] for s in json.loads((ROOT/'frozen-plan.json').read_text())['test']])
    (ROOT/'final-selection.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('frozen_utc','selected_checkpoint_sha256','scene_plan_sha256')},indent=2))


if __name__=='__main__':main()
