"""Consumed V24 trace replay: first changed decision only, not new task outcomes."""
import hashlib
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'runtime/src'))
from pour_source_watchdog import SourceWatchdog,DEFAULT


def main():
    result=json.loads((PROJECT/'evidence/finish-A/final-benchmark.json').read_text())
    audit_path=PROJECT/'evidence/finish-A/recording-integrity.json'
    audit=json.loads(audit_path.read_text())
    sidecars={r['relative_path']:r for r in audit['sidecar_files']}
    passed={r['name']:r['strict_pass'] for r in result['cases']}
    cases=[]
    for path in sorted((PROJECT/'data/material-A-finish').glob('*/*FN30_A_test_*/policy-decisions.json')):
        for checked in (path,path.parent/'parameters.json'):
            identity=sidecars[str(checked.relative_to(PROJECT/'data/material-A-finish'))]
            assert checked.stat().st_size==identity['bytes']
            assert hashlib.sha256(checked.read_bytes()).hexdigest()==identity['sha256']
        spec=json.loads((path.parent/'parameters.json').read_text())
        rows=json.loads(path.read_text());w=SourceWatchdog(DEFAULT);first=None;stop=None
        for row in rows:
            if row.get('stop_latched'):
                stop=dict(t=row['t'],reason=row['stop_latched']);break
            view=row.get('empty_views') or {}
            if not row.get('visual_grasp_confirmed') or 'encoder_hold' not in view:continue
            reason=w.update(row['t'],held=view['encoder_hold'],present=row['source_present_probability'],
                empty=row['empty_probability'],tilt=row['estimated_tilt_from_joints'],phase=row['own_pour_phase_deg'],
                cap=131.,flow_ml_s=row['visual_flow_fraction_s']*72.8)
            if reason and first is None:first=dict(t=row['t'],reason=reason)
        cases.append(dict(name=spec['name'],decisions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            old_strict_pass=passed[spec['name']],old_stop=stop,new_earlier_recovery=first))
    assert len(cases)==30
    regressions=[r['name'] for r in cases if r['old_strict_pass'] and r['new_earlier_recovery']]
    report=dict(scope='All 30 V24 test traces are now explicitly consumed for follow-up controller development. Frozen original scores unchanged. Replay can establish first altered decision only; subsequent contact, fluid, return and success require native runs.',
        config=DEFAULT,cases=cases,earlier_abort_on_old_successes=regressions,
        prior_integrity_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),verified_sidecars=60,
        source_sha256=hashlib.sha256((PROJECT/'runtime/src/pour_source_watchdog.py').read_bytes()).hexdigest())
    out=PROJECT/'evidence/size-A';out.mkdir(exist_ok=True)
    (out/'source-watchdog-replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),earlier_abort_on_old_successes=regressions,
        changed=[r for r in cases if r['new_earlier_recovery']])))
    assert not regressions


if __name__=='__main__':main()
