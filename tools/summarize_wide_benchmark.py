"""Recompute wide-test outcomes from verified native backups before delivery."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[1];E=PROJECT/'evidence/wide-A';DATA=PROJECT/'data/material-A-wide'
sys.path.insert(0,str(PROJECT/'runtime/src'))
from pour_wide_report import strict_success,reasons


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    report=read(E/'wide-results.json');selection=read(E/'final-selection.json');plan=read(E/'frozen-plan.json')
    model=read(E/'model-summary.json');integrity=read(E/'recording-integrity.json')
    checkpoint=sha(E/'best.pt')
    assert checkpoint==selection['selected_checkpoint_sha256']==model['checkpoint_sha256']==report['checkpoint_sha256']
    assert sha(E/'frozen-plan.json')==selection['scene_plan_sha256']==report['scene_plan_sha256']==model['scene_plan_sha256']
    assert model['sources']==selection['runtime_sources']
    for n,digest in selection['runtime_sources'].items():
        assert sha(PROJECT/'runtime/src'/n)==digest,n
        committed=subprocess.check_output(['git','show',selection['source_commit']+':runtime/src/'+n],cwd=PROJECT)
        assert hashlib.sha256(committed).hexdigest()==digest,(n,'Source commit differs')
    assert report['status']=='complete' and len(report['cases'])==30
    expected={s['name']:s for s in plan['test']};assert {r['name'] for r in report['cases']}==set(expected)
    assert not integrity['pending_attempts'] and integrity['native_checks_passed']
    assert {r['name'] for r in integrity['files']}=={r['name'] for r in report['all_attempts']}
    indexed={r['name']:r for r in integrity['files']}
    for r in integrity['files']+integrity['sidecar_files']:
        p=DATA/r['relative_path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    for r in report['cases']:
        folder=(DATA/indexed[r['name']]['relative_path']).parent;spec=read(folder/'parameters.json')
        assert all(spec[k]==v for k,v in expected[r['name']].items())
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
    final=snapshots['runtime-final']['health'];assert not final['queue'] and not final['recording']['active'] and not final['error'] and not final['learning_error']
    manifest=read(E/'data-manifest.json');assert sha(E/'data-manifest.json')==model['data_manifest_sha256']
    consumed={s['name']+'_r0' for s in plan['development']}
    assert all(not x['name'].startswith('XW30_A_test') and (not x['name'].startswith('XW30_A_dev') or x['name'] in consumed) for x in manifest)
    dev_level=[x for x in manifest if x['name'] in consumed]
    assert {x['name'] for x in dev_level}==consumed and len(dev_level)==model['training']['consumed_development_level_episodes']
    assert all(x['split']=='train' and x['recording_split']=='development' and x['consumed_development'] for x in dev_level)
    train={x['name'] for x in manifest if x['split']=='train'};valid={x['name'] for x in manifest if x['split']=='validation'}
    assert train.isdisjoint(valid) and len(train)==model['training']['train_episodes'] and len(valid)==model['training']['validation_episodes']
    accepted={x['name'] for x in report['demonstrations'] if x['reference_accepted']}
    assert {x['reference_case'] for x in report['demonstrations'] if x['reference_accepted']}=={x['name'] for x in plan['demonstrations']}
    assert all(not x['name'].startswith('T30_A_demo_') or x['name'] in accepted for x in manifest)
    for key in ('train_episodes','validation_episodes'):
        assert all(not n.startswith(('XW30_A_test','XW30_A_dev')) and (not n.startswith('T30_A_demo_') or n in accepted) for n in model['motion'][key])
    calibration=read(E/'localization-development-manifest.json')
    assert sha(E/'localization-development-manifest.json')==model['localization_development_manifest_sha256']
    expected_dev={s['name']+'_r0' for s in plan['development']}
    assert {r['name'] for r in calibration['records']}==expected_dev==set(model['training']['localization_development_names'])
    for r in calibration['records']:assert r['sha256']==indexed[r['name']]['sha256']
    groups={}
    for reason in ('cup_near_full','bottle_empty'):
        rows=[r for r in report['cases'] if r['expected_stop']==reason];assert len(rows)==15
        groups[reason]=dict(attempts=15,strict_passes=sum(r['strict_pass'] for r in rows))
    final_report=dict(**report,groups=groups,verified_native_backups=len(integrity['files']),
        verified_sidecars=len(integrity['sidecar_files']),independently_recomputed_primary_scores=30)
    (E/'final-benchmark.json').write_text(json.dumps(final_report,indent=2)+'\n')
    (E/'runtime-comparison.json').write_text(json.dumps(dict(same_container_caps_sources_checkpoint_asset=True,
        source_count=len(selection['runtime_sources']),snapshots=snapshots,
        scope='Read-only snapshots and identity-bound guard. No foreign job was signaled or reconfigured; snapshots cannot prove uninterrupted foreign performance.'),indent=2)+'\n')
    print(json.dumps(dict(counts=report['counts'],groups=groups,backups=len(integrity['files']),sources=len(selection['runtime_sources'])),indent=2))


if __name__=='__main__':main()
