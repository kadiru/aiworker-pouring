"""Freeze a validated corrective controller before the new thirty-case test."""
import argparse,datetime,hashlib,json,os,urllib.request
from pathlib import Path
from pour_reliable_study import ROOT,PREFIX,plan,RUNTIME_SOURCES
from pour_reliable_report import report


def validate_development(rows, known_spill_particles=0):
    """Readiness for an honest evaluation; never changes a trial's score."""
    assert known_spill_particles >= 0
    for x in rows:
        goal=(x['cup_surface_height_fraction']>=.90 if x['expected_stop']=='cup_near_full'
            else x['source_remaining_ml']<=max(.25,.01*x['source_initial_ml']))
        assert (x['grasp_verified'] and x['visual_grasp_confirmed'] and
            x['sustained_grasp_fraction']>=.8 and x['completed_return_release_sequence'] and
            x['policy_completed'] and x['stop_reason']==x['expected_stop'] and goal and not x['gates']), x['name']
    failures=[x for x in rows if not x['surface_goal_zero_spill_success']]
    for x in failures:
        assert known_spill_particles and not x['zero_spill'], 'Strict development failure retained; explicit readiness allowance required'
        assert max(x['final_outside_ml'],x['ever_below_worktop_ml']) <= known_spill_particles*.0018**3*1e6+1e-10, x['name']
    return failures


def main(known_spill_particles=0):
    assert not list(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}test*/parameters.json'))
    r=report();p=plan();chosen={}
    for row in r['development']:chosen[row['development_case']]=row
    assert set(chosen)=={x['development_case'] for x in p['development']}
    failures=validate_development(list(chosen.values()),known_spill_particles)
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    sources={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in RUNTIME_SOURCES}
    summary=json.loads((ROOT/'model-summary.json').read_text())
    assert summary['checkpoint_sha256']==r['checkpoint_sha256'] and summary['sources']==sources
    frozen=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=r['checkpoint_sha256'],
        scene_plan_sha256=r['scene_plan_sha256'],runtime_sources=sources,source_commit=os.environ['AIWORKER_SOURCE_COMMIT'],
        development=list(chosen.values()),all_development_attempts=r['all_attempts'],incomplete_development_attempts=r['incomplete_attempts'],expected_test_names=[x['name'] for x in p['test']],
        documented_development_failures=failures,development_strict_successes=len(chosen)-len(failures),
        known_spill_particles_allowed_for_readiness_only=known_spill_particles,
        selection_rationale=('Localization, settled-fill/empty stop and return conditions passed all development scenes. Retain the documented small spill failure and measure residual spill risk in the unchanged strict thirty-case evaluation.' if failures else 'All development scenes met the unchanged strict goal.'),
        evaluation_scoring_changed=False,
        test_policy=p['test_policy'],primary_definition=r['primary_definition'],legacy_definition=r['legacy_definition'],volume_units=r['volume_units'])
    (ROOT/'final-selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({k:frozen[k] for k in ('frozen_utc','selected_checkpoint_sha256','scene_plan_sha256','source_commit')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--document-known-spill-particles',type=int,default=0,
        help='Explicit readiness allowance only; failed development scores and strict test scoring remain unchanged')
    a=p.parse_args();main(a.document_known_spill_particles)
