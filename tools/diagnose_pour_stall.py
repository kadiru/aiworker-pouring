"""CPU-only analysis of saved commands and measured joints; no simulator calls."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/learning-A'


def read(path):
    return json.loads(path.read_text())


def diagnose():
    cases = []
    run = ROOT / 'data/material-A-evaluation/20261102T232206Z'
    for path in sorted(run.glob('*test*/samples.json')):
        rows = read(path)
        metrics = read(path.parent / 'metrics.json')
        # Match the commanded target with its following measured state, as recorded.
        late = [r for r in rows if r['t'] >= 22.]
        result = dict(case=path.parent.name, grasp_verified=metrics['grasp_verified'],
                      source=str(path), window_start_sim_s=22., samples=len(late))
        if late:
            errors = [r['joint_targets_deg'][j] - r['joint_positions_deg'][j]
                      for r in late for j in range(7)]
            spans = [max(r['joint_targets_deg'][j] for r in late) -
                     min(r['joint_targets_deg'][j] for r in late) for j in range(7)]
            decisions = [r for r in read(path.parent / 'policy-decisions.json')
                         if r['t'] >= 22.]
            result.update(arm_tracking_rmse_deg=math.sqrt(sum(e*e for e in errors)/len(errors)),
                          arm_tracking_max_error_deg=max(map(abs, errors)),
                          max_per_joint_command_span_deg=max(spans),
                          bottle_tilt_min_deg=min(r['bottle_tilt_deg'] for r in late),
                          bottle_tilt_max_deg=max(r['bottle_tilt_deg'] for r in late),
                          command_clamp_count=sum(r['delta_clamped'] for r in decisions),
                          decision_samples=len(decisions),
                          cup_inventory_change_ml=late[-1]['cup_ml']-late[0]['cup_ml'],
                          source_inventory_change_ml=late[-1]['bottle_ml']-late[0]['bottle_ml'])
        cases.append(result)

    manifest = read(EVIDENCE / 'dataset-manifest.json')
    admitted = {r['name'] for r in manifest if r.get('accepted') and
                r.get('split') == 'train' and r.get('task_scope', 'full_pour') == 'full_pour'}
    references = []
    for path in sorted((ROOT/'data/material-A-study').glob('*/*/samples.json')):
        spec = read(path.parent / 'parameters.json')
        if spec['name'] not in admitted:
            continue
        rows = read(path)
        first = next((r for r in rows if r['cup_ml']-rows[0]['cup_ml'] >= .5), None)
        references.append(dict(case=spec['name'], source=str(path),
                               first_half_ml_time_sim_s=first['t'] if first else None,
                               first_half_ml_tilt_deg=first['bottle_tilt_deg'] if first else None))

    report = dict(evidence_kind='CPU analysis of native saved rollouts; no new training or simulation',
                  tracking_units='degrees; seven arm joints only, fingers intentionally excluded',
                  volume_units='nominal particle volume',
                  limitation='Plateau is in issued learned-controller targets. This does not isolate the neural model from temporal ensembling or establish a universal flow angle.',
                  test_cases=cases, admitted_full_training_references=references,
                  conclusion='Five held-bottle tests track arm targets closely, without late command clamping or fluid transfer. Correct the learned lift-to-pour progression.',
                  next_action='Collect full expert recoveries from the actual held/tilted policy states, preserving alignment, flow, stopping and set-down.')
    output = EVIDENCE / 'stall-diagnostic.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    diagnose()
