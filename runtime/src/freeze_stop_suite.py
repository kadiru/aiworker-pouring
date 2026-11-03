"""Freeze exact model/config/scene identity before queuing six fresh tests."""
import datetime,hashlib,json
from pathlib import Path
from pour_stop_study import ROOT


def main():
    outputs=Path('/workspace/outputs/fountain')
    assert not list(outputs.glob('*/*S_A_test*/parameters.json')),'Fresh tests already exist; refusing a second freeze'
    assert not list(outputs.glob('*/*S_A_*/episode.partial.hdf5')),'A development recording is still active'
    attempts=[]
    for p in sorted(outputs.glob('*/*S_A_*/metrics.json')):
        m=json.loads(p.read_text());s=json.loads((p.parent/'parameters.json').read_text())
        attempts.append(dict(name=s['name'],path=str(p.parent),checkpoint_sha256=m['policy']['checkpoint_sha256'],
            surface_goal_success=m.get('surface_task_success_zero_spill',False),legacy_success=m.get('task_success_zero_spill',False),
            final_outside_ml=m['spill']['final_outside_ml'],source_remaining_ml=m['bottle_result']['bottle_ml'],
            cup_surface_fraction=m['bottle_result']['cup_fill_fraction'],policy_completed=m['policy']['done']))
    sha=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest();selected=[r for r in attempts if r['checkpoint_sha256']==sha]
    assert selected,'Selected candidate has no completed native development replay'
    note=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=sha,
        scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),development_attempts=attempts,
        selection_reason='V9 replaces drift-prone pour increments, completes an empty-source zero-spill development task, and uses a native-calibrated anticipatory full stop; all prior failures retained.',
        caveat='Empty-source development used the preceding identical-weights policy with full-floor.82; pre-test calibration changes only full-stop lead. Exact-final-checkpoint full replay is listed above.',
        full_lead_calibration=json.loads((ROOT/'full-lead-calibration.json').read_text()),
        test_policy='Run all six frozen cases once. No further model, configuration, actor or physics changes during the block. Report every outcome under both surface-height and original nominal-inventory scoring.',
        expected_test_names=[s['name'] for s in json.loads((ROOT/'frozen-plan.json').read_text())['test']])
    (ROOT/'final-selection.json').write_text(json.dumps(note,indent=2)+'\n')
    print(json.dumps(dict(frozen_utc=note['frozen_utc'],selected_checkpoint_sha256=sha,development_cases=len(attempts)),indent=2))


if __name__=='__main__':main()
