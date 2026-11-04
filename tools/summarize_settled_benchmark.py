"""Verify frozen identities and native backup hashes before final reporting."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/settled-A';DATA=ROOT/'data/material-A-settled'


def read(p):return json.loads(p.read_text())


def main():
    r=read(E/'settled-results.json');s=read(E/'final-selection.json');before=read(E/'runtime-before-tests.json');after=read(E/'runtime-final.json')
    assert r['status']=='complete' and len(r['cases'])==30
    sha=hashlib.sha256((E/'best.pt').read_bytes()).hexdigest()
    assert sha==s['selected_checkpoint_sha256']==r['checkpoint_sha256']==before['selected_checkpoint_sha256']==after['selected_checkpoint_sha256']
    assert hashlib.sha256((E/'frozen-plan.json').read_bytes()).hexdigest()==s['scene_plan_sha256']==r['scene_plan_sha256']
    assert before['sources']==after['sources']==s['runtime_sources']
    for n,h in s['runtime_sources'].items():assert hashlib.sha256((ROOT/'runtime/src'/n).read_bytes()).hexdigest()==h,n
    assert before['container']==after['container'] and before['limits']==after['limits'] and after['guard_processes']
    assert not after['health']['queue'] and not after['health']['recording']['active'] and not after['health']['error'] and not after['health']['learning_error']
    assert before['asset_sha256']==after['asset_sha256']=='82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d'
    expected={x['name'] for x in read(E/'frozen-plan.json')['test']};assert {x['name'] for x in r['cases']}==expected
    assert all(x['checkpoint_sha256']==sha and not x['teacher_fallback'] for x in r['cases'])
    # Independently check the declared whole-task score against saved evidence.
    # A metric flag alone cannot credit a wrong stop, missing transfer or return.
    for x in r['cases']:
        zero=x['final_outside_ml']==0 and x['ever_below_worktop_ml']==0
        goal=(x['cup_surface_height_fraction']>=.90 if x['expected_stop']=='cup_near_full'
              else x['source_remaining_ml']<=max(.25,.01*x['source_initial_ml']))
        primary=bool(x['grasp_verified'] and x['visual_grasp_confirmed'] and
            x['sustained_grasp_fraction']>=.8 and x['delivered_to_cup_ml']>=1 and
            x['policy_completed'] and x['completed_return_release_sequence'] and
            x['stop_reason']==x['expected_stop'] and not x['gates'] and goal and zero)
        assert x['zero_spill']==zero and x['surface_goal_zero_spill_success']==primary, x['name']
    manifest=read(E/'data-manifest.json')
    assert all(not x['name'].startswith('L30_A_') for x in manifest)
    train={x['name'] for x in manifest if x['split']=='train'};val={x['name'] for x in manifest if x['split']=='validation'}
    metadata=read(E/'model-summary.json');training=metadata['training']
    assert metadata['checkpoint_sha256']==sha and metadata['sources']==s['runtime_sources']
    assert hashlib.sha256((E/'data-manifest.json').read_bytes()).hexdigest()==metadata['data_manifest_sha256']
    assert not train & val and len(train)==training['train_episodes'] and len(val)==training['validation_episodes']
    integrity=read(E/'recording-integrity.json');assert len(integrity['files'])==len(r['all_attempts'])
    assert len(integrity['sidecar_files'])==4*len(integrity['files']) and {x['name'] for x in integrity['files']}=={x['name'] for x in r['all_attempts']}
    assert len(integrity['partial_files'])==len(r['incomplete_attempts'])
    for x in integrity['files']+integrity['sidecar_files']+integrity['partial_files']+integrity['partial_sidecar_files']:
        p=DATA/x['relative_path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
    indexed={x['relative_path']:x for k in ('files','sidecar_files','partial_files','partial_sidecar_files') for x in integrity[k]}
    for archive in E.glob('round*-development-integrity.json'):
        old=read(archive)
        for k in ('files','sidecar_files','partial_files','partial_sidecar_files'):
            for x in old[k]:
                current=indexed[x['relative_path']]
                assert (current['sha256'],current['bytes'])==(x['sha256'],x['bytes']),str(archive)
    assert all(x['valid'] for x in integrity['files'])
    (DATA/'backup-index.json').write_text(json.dumps(dict(all_native_checksums_match=True,**integrity),indent=2)+'\n')
    (E/'final-benchmark.json').write_text(json.dumps(dict(**r,verified_native_backups=len(integrity['files']),verified_sidecars=len(integrity['sidecar_files']),independently_recomputed_primary_scores=30),indent=2)+'\n')
    print(json.dumps(dict(counts=r['counts'],backups=len(integrity['files']),sidecars=len(integrity['sidecar_files']),source_hashes=len(s['runtime_sources'])),indent=2))


if __name__=='__main__':main()
