"""Explain frozen once-only scores from stored predictions; never rerun inference."""
import collections,json,math
from pathlib import Path
import torch
from fit_dynamic_rim import ROOT,read,sha


def main():
    result=read(ROOT/'moving-holdout-evaluation.json');selection=read(ROOT/'component-selection.json')
    assert len(result['cases'])==18 and result['selection_sha256']==sha(ROOT/'component-selection.json')
    assert all(sha(ROOT/n)==h for n,h in selection['frozen_files'].items())
    loc=torch.load(ROOT/selection['locator'],map_location='cpu',weights_only=False)
    g=read(ROOT/'protocol.json')['gates'];threshold=loc['threshold'];rows=[]
    def level_ok(m):
        return m['valid'] and m['mae']<g['level_validation_mae'] and (m['near_max'] is None or (m['near_max']<g['level_validation_near_max_error'] and m['near_overestimate']<g['level_validation_near_overestimate']))
    for r in result['cases']:
        frames=r['frames'];negative=r['synthetic_absent_max'];minimum=min(f['score'] for f in frames)
        checks=dict(physical=r['physical'],complete=len(frames)==40 and all(f['color_valid'] for f in frames),
                    tracker_status_parity=r['tracker_status_parity'],raw_localization=max(r['initial_raw_errors_mm'])<g['localization_max_error_mm'],
                    anchor_localization=r['anchor_error_mm'] is not None and r['anchor_error_mm']<g['localization_max_error_mm'],
                    tracked_localization=len(frames)==40 and all(f['localization_valid'] and f['localization_error_mm'] is not None and math.isfinite(f['localization_error_mm']) and f['localization_error_mm']<g['localization_max_error_mm'] and math.isfinite(f['score']) and f['score']>=threshold for f in frames),
                    rgb_level=level_ok(r['rgb_metrics']),color_level=level_ok(r['color_metrics']),
                    independent_agreement=r['max_disagreement']<g['crosscheck_max_disagreement'],
                    locator_cpu_gpu_parity=r['locator_cpu_gpu_max_pixels']<.1,
                    level_cpu_gpu_parity=all(f['level_parity'] is not None and f['level_parity']<g['cpu_gpu_level_difference'] for f in frames),
                    absent_cup_rejection=negative<threshold,absence_margin=minimum-negative>g['localization_negative_margin'])
        assert all(checks.values())==r['passed'],'Explanation must reproduce the original result'
        rows.append(dict(name=r['name'],passed=r['passed'],checks=checks,failed_checks=[k for k,v in checks.items() if not v],
                         raw_max_mm=max(r['initial_raw_errors_mm']),tracked_max_mm=max((f['localization_error_mm'] for f in frames if f['localization_error_mm'] is not None),default=None),
                         localized_frames=sum(f['localization_valid'] for f in frames),
                         rgb_metrics=r['rgb_metrics'],color_metrics=r['color_metrics'],max_disagreement=r['max_disagreement'],
                         synthetic_absent_max=negative,minimum_present_score=minimum,negative_margin=minimum-negative))
    frames=[f for r in result['cases'] for f in r['frames']]
    aggregate={}
    for branch in ('rgb','color'):
        valid=[f for f in frames if f[branch] is not None and math.isfinite(f[branch])]
        errors=[f[branch]-f['target'] for f in valid];near=[e for f,e in zip(valid,errors) if f['target']>.85]
        aggregate[branch]=dict(finite_frames=len(valid),total_frames=len(frames),mae=sum(map(abs,errors))/len(errors),max_abs=max(map(abs,errors)),near_max=max(map(abs,near)),near_overestimate=max(near),worst_episode_mae=max((r[branch+'_metrics']['mae'] for r in rows if r[branch+'_metrics']['valid']),default=None))
    report=dict(passed=result['passed'],passed_cases=result['passed_cases'],total_cases=len(rows),motion_frames=len(frames),
                gate_pass_counts={k:sum(r['checks'][k] for r in rows) for k in rows[0]['checks']},
                failure_counts=dict(collections.Counter(k for r in rows for k in r['failed_checks'])),
                aggregate=aggregate,raw_initial_max_mm=max(r['raw_max_mm'] for r in rows),tracked_max_mm=max(r['tracked_max_mm'] for r in rows if r['tracked_max_mm'] is not None),
                max_disagreement=max(r['max_disagreement'] for r in rows if r['max_disagreement'] is not None),cpu_gpu_level_max=max(f['level_parity'] for f in frames if f['level_parity'] is not None),threshold=threshold,
                cases=rows,evaluation_sha256=sha(ROOT/'moving-holdout-evaluation.json'),selection_sha256=sha(ROOT/'component-selection.json'),
                helper_sha256=sha(Path(__file__)),scope='Read-only explanation of original frozen scores. No inference rerun, fitting, thresholds changed, retest, or active controller update.',autonomy_accepted=False)
    with (ROOT/'reserved-gate-breakdown.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
