"""Independently recompute all thirty outcomes from audited scalar sidecars."""
import hashlib,json,subprocess,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];E=PROJECT/'evidence/finish-A';DATA=PROJECT/'data/material-A-finish'
sys.path.insert(0,str(PROJECT/'runtime/src'))
from pour_wide_report import strict_success,reasons
from finish_media_source import sha

def read(p):return json.loads(p.read_text())

def main():
    report=read(E/'finish-results.json');selection=read(E/'final-selection.json');plan=read(E/'frozen-plan.json')
    model=read(E/'model-summary.json');integrity=read(E/'recording-integrity.json')
    checkpoint=sha(E/'best.pt')
    assert checkpoint==selection['selected_checkpoint_sha256']==model['checkpoint_sha256']==report['checkpoint_sha256']
    assert sha(E/'frozen-plan.json')==selection['scene_plan_sha256']==report['scene_plan_sha256']==model['scene_plan_sha256']
    assert model.get('runtime_sources',model['sources'])==selection['runtime_sources']
    for n,digest in selection['runtime_sources'].items():
        assert sha(PROJECT/'runtime/src'/n)==digest,n
        committed=subprocess.check_output(['git','show',selection['source_commit']+':runtime/src/'+n],cwd=PROJECT)
        assert hashlib.sha256(committed).hexdigest()==digest,(n,'Source commit differs')
    assert report['status']=='complete' and len(report['cases'])==30
    expected={s['name']:s for s in plan['test']};assert {r['name'] for r in report['cases']}==set(expected)
    assert not integrity['pending_attempts'] and integrity['native_checks_passed'] and integrity['scope']=='evaluation'
    indexed={r['name']:r for r in integrity['files']}
    assert set(expected)<=set(indexed)
    for identity in integrity['sidecar_files']+integrity.get('partial_sidecar_files',[]):
        path=DATA/identity['relative_path'];assert path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256']
    for r in report['cases']:
        folder=(DATA/indexed[r['name']]['relative_path']).parent;spec=read(folder/'parameters.json')
        assert spec==expected[r['name']] and spec['protocol']==18
        samples=read(folder/'samples.json');decisions=read(folder/'policy-decisions.json');m=read(folder/'metrics.json');last=samples[-1]
        assert m['policy']['checkpoint_sha256']==checkpoint and not m['policy']['teacher_fallback']
        assert r['cup_surface_height_fraction']==last['cup_fill_fraction'] and r['source_remaining_ml']==last['bottle_ml']
        assert r['final_outside_ml']==last['spilled_ml']+last['airborne_ml']
        assert r['ever_below_worktop_ml']==max(x.get('ever_table_ml',0.) for x in samples)
        assert r['zero_spill']==(r['final_outside_ml']==r['ever_below_worktop_ml']==0)
        tipping=[x for x in samples if abs(x['bottle_tilt_deg'])>20.]
        assert r['sustained_grasp_fraction']==sum(x['both_fingers'] for x in tipping)/max(1,len(tipping))
        assert r['grasp_verified']==any(x['grasp_verified'] for x in samples)
        assert r['visual_grasp_confirmed']==any(x.get('visual_grasp_confirmed',False) for x in decisions)
        order=('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose')
        times=[next((x['t'] for x in decisions if x['stop_latched'] and x['mode']==mode),None) for mode in order]
        ordered=all(t is not None for t in times) and all(a<b for a,b in zip(times,times[1:]))
        assert r['completed_return_release_sequence']==bool(ordered and m['policy']['done'])
        assert r['strict_pass']==strict_success(r)==(not reasons(r))
        if r['strict_pass']:assert abs(last['bottle_tilt_deg'])<8 and abs(last['bottle_base_height_m'])<=.005
    snapshots={p.stem:read(p) for p in [E/'runtime-before-tests.json',*sorted(E.glob('runtime-after-test*.json')),E/'runtime-final.json']}
    before=snapshots['runtime-before-tests']
    for label,s in snapshots.items():
        assert s['selected_checkpoint_sha256']==checkpoint and s['sources']==selection['runtime_sources'] and s['guard_processes']
        for key in ('container','limits','asset_sha256'):assert s[key]==before[key],(label,key)
    assert before['container']==selection['container']
    assert before['asset_sha256']=='82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d'
    final=snapshots['runtime-final']['health'];assert not final['queue'] and not final['recording']['active'] and not final.get('error') and not final.get('learning_error')
    manifest=read(E/'data-manifest.json');assert sha(E/'data-manifest.json')==model['data_manifest_sha256']
    assert sha(E/'calibration-plan.json')==model['calibration_plan_sha256']
    records=manifest['consumed_historical_records'];assert len({r['name'] for r in records})==len(records)
    assert all(not r['name'].startswith(('FN30_A_dev_','FN30_A_test_')) for r in records)
    train={r['name'] for r in records if r['split']=='train'};valid={r['name'] for r in records if r['split']=='validation'}
    assert train.isdisjoint(valid)
    calibrations={s['name'] for s in read(E/'calibration-plan.json')['cases']}
    assert {r['name'] for r in records if r['name'].startswith('FCAL_A_')}==calibrations
    groups={}
    for reason in ('cup_near_full','bottle_empty'):
        rows=[r for r in report['cases'] if r['expected_stop']==reason];assert len(rows)==15
        groups[reason]=dict(attempts=15,strict_passes=sum(r['strict_pass'] for r in rows))
    final_report=dict(**report,groups=groups,native_audited_recordings=len(integrity['files']),
        retained_incomplete_recordings=integrity.get('retained_incomplete_attempts',[]),
        recording_repair=model.get('recording_repair'),
        full_stop_calibration=model.get('full_stop_calibration'),
        initial_pour_calibration=model.get('initial_pour_calibration'),
        verified_local_sidecars=len(integrity['sidecar_files']),independently_recomputed_primary_scores=30,
        recording_storage='Original HDF5 recordings remain on Ubuntu; each video source is fetched temporarily and byte/SHA-verified. No complete local raw-data backup is claimed.')
    (E/'final-benchmark.json').write_text(json.dumps(final_report,indent=2)+'\n')
    (E/'runtime-comparison.json').write_text(json.dumps(dict(same_container_caps_sources_checkpoint_asset=True,
        source_count=len(selection['runtime_sources']),snapshots=snapshots,
        scope='No foreign job was signaled or reconfigured; point-in-time snapshots do not prove zero shared-host overhead.'),indent=2)+'\n')
    print(json.dumps(dict(counts=report['counts'],groups=groups,sidecars=len(integrity['sidecar_files'])),indent=2))

if __name__=='__main__':main()
