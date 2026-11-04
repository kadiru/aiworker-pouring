"""Freeze a validated corrective controller before the new thirty-case test."""
import datetime,hashlib,json,os,urllib.request
from pathlib import Path
from pour_settled_study import ROOT,PREFIX,plan,RUNTIME_SOURCES
from pour_settled_report import report


def main():
    assert not list(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}test*/parameters.json'))
    r=report();p=plan();chosen={}
    for row in r['development']:chosen[row['development_case']]=row
    assert set(chosen)=={x['development_case'] for x in p['development']}
    assert all(x['surface_goal_zero_spill_success'] and x['completed_return_release_sequence'] for x in chosen.values()),'Resolve development failures first'
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    sources={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in RUNTIME_SOURCES}
    summary=json.loads((ROOT/'model-summary.json').read_text())
    assert summary['checkpoint_sha256']==r['checkpoint_sha256'] and summary['sources']==sources
    frozen=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=r['checkpoint_sha256'],
        scene_plan_sha256=r['scene_plan_sha256'],runtime_sources=sources,source_commit=os.environ['AIWORKER_SOURCE_COMMIT'],
        development=list(chosen.values()),all_development_attempts=r['all_attempts'],expected_test_names=[x['name'] for x in p['test']],
        test_policy=p['test_policy'],primary_definition=r['primary_definition'],legacy_definition=r['legacy_definition'],volume_units=r['volume_units'])
    (ROOT/'final-selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({k:frozen[k] for k in ('frozen_utc','selected_checkpoint_sha256','scene_plan_sha256','source_commit')},indent=2))


if __name__=='__main__':main()
