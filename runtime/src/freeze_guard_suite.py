"""Freeze model/config/source identity before exactly six new native tests."""
import datetime,hashlib,json
from pathlib import Path
from pour_guard_study import ROOT


def main():
    base=Path('/workspace/outputs/fountain');assert not list(base.glob('*/*G_A_test*/parameters.json')),'Tests already started'
    sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest();results=json.loads((ROOT/'guard-results.json').read_text())
    assert results['checkpoint_sha256']==sha
    development=results['groups']['regression']['rows']+results['groups']['validation']['rows']
    assert len(development)==4,'Complete both regressions and both new validation cases first'
    r=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=sha,
        scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        selection_reason='New visual/proprioceptive grasp gate and settled-level fill feedback, selected using development only; all prior failed/aborted development retained.',
        development=development,all_development_attempts=results['all_attempts'],
        test_policy='Run six new seed20261122 cases once; no model/config/controller/physics changes, no expert fallback during the block.',
        deadline_note='40 simulated seconds and 12-second grasp deadline allow one retry and settled top-up cycles; success criteria and accepted physics unchanged.',
        expected_test_names=[s['name'] for s in json.loads((ROOT/'frozen-plan.json').read_text())['test']])
    (ROOT/'final-selection.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('frozen_utc','selected_checkpoint_sha256','scene_plan_sha256')},indent=2))


if __name__=='__main__':main()
