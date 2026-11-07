"""Freeze one reviewed finish candidate after eight completed development cases.

Explicit invocation only. Archives model/source, snapshots isolation, and queues
the thirty unused reserved scenes once. Never trains or repairs a test case.
"""
import argparse,datetime,hashlib,json,pathlib,subprocess,sys
from watch_wide_collection import isolation,health,command,CID,STUDY,HOST_ROOT,GPU
from watch_finish_trials import capture
STUDY=HOST_ROOT/'learning/A-finish-20261106-v1'
sys.path.insert(0,str(HOST_ROOT/'src'))
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons
from wide_release_gate import validate_development_outcome


def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--review',required=True);parser.add_argument('--round',type=int,default=0)
    parser.add_argument('--withdrawal',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--evaluate-known-full-stop-limitations',metavar='RATIONALE')
    a=parser.parse_args()
    destination=pathlib.Path(a.output_dir);assert not destination.exists()
    manifest=read(a.manifest);review=read(a.review);model=read(STUDY/'model-summary.json');plan=read(STUDY/'frozen-plan.json')
    assert not (STUDY/'final-selection.json').exists()
    assert not list((HOST_ROOT/'outputs/fountain').glob('*/*FN30_A_test*/parameters.json'))
    checkpoint=sha(STUDY/'best.pt');assert checkpoint==model['checkpoint_sha256']
    assert sha(STUDY/'frozen-plan.json')==model['scene_plan_sha256']
    assert '4070' in model['device']
    assert model['terminal_drain_hold_s']==0.
    runtime_sources=model.get('runtime_sources',model['sources'])
    for n,digest in runtime_sources.items():assert manifest['files'][n]==sha(HOST_ROOT/'src'/n)==digest,n
    expected={model.get('development_aliases',{}).get(s['name']+f'_r{a.round}',s['name']+f'_r{a.round}'):s for s in plan['development']};assert len(expected)==8
    rows=[];seen=set()
    for p in sorted((HOST_ROOT/'outputs/fountain').glob('*/*FN30_A_dev_*/parameters.json')):
        s=read(p);name=s['name']
        if name not in expected:continue
        assert name not in seen;seen.add(name)
        assert all(s[k]==v for k,v in expected[name].items() if k!='name')
        assert s['protocol']==18 and s['development_round']==a.round
        assert (p.parent/'episode.hdf5').exists() and not (p.parent/'episode.partial.hdf5').exists()
        row=completed_row(p.parent,checkpoint);passed=strict_success(row);failure_reasons=reasons(row)
        validate_development_outcome(row,passed,failure_reasons,a.evaluate_known_full_stop_limitations)
        decisions=read(p.parent/'policy-decisions.json');samples=read(p.parent/'samples.json')
        modes=('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose')
        times=[next(x['t'] for x in decisions if x['stop_latched'] and x['mode']==mode) for mode in modes]
        assert all(x<y for x,y in zip(times,times[1:]))
        assert abs(samples[-1]['bottle_tilt_deg'])<8 and abs(samples[-1]['bottle_base_height_m'])<=.005
        rows.append(dict(row,strict_pass=passed,failure_reasons=failure_reasons))
    assert seen==set(expected)
    assert review['checkpoint_sha256']==checkpoint and review['manual_review']=='inspected'
    assert {r['name'] for r in review['cases']}==seen and all(r['manual_review']=='inspected' for r in review['cases'])
    integrity=read(STUDY/'development-recording-integrity.json')
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    indexed={r['name']:r for r in integrity['files']}
    for r in review['cases']:assert r['source_sha256']==indexed[r['name']]['sha256']
    withdrawal=read(a.withdrawal);motion={r['name']:r for r in withdrawal['cases']}
    assert set(motion)==seen
    for name,r in motion.items():
        assert r['source_sha256']==indexed[name]['sha256']
        assert r['post_release_gripper_contact_frames']==0,(name,'Post-release contact')
        assert r['longest_sampled_upright_near_stationary_over_cup_s']<=.35,(name,'Lingering above cup')
        starts=r['stage_starts_s'];duration=starts['Setting bottle down']-starts['Returning upright']
        assert abs(duration-3.1)<=.051,(name,'Unexpected return duration',duration)
    witness=isolation();h=health()
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    own={int(x.strip()) for x in subprocess.check_output(['docker','top',CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in witness['gpu_processes'].splitlines() if r.startswith(GPU))
    selection=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_checkpoint_sha256=checkpoint,
        scene_plan_sha256=model['scene_plan_sha256'],runtime_sources=runtime_sources,source_commit=manifest['source_commit'],
        fitted_sources=model['sources'],recording_repair=model.get('recording_repair'),
        full_stop_calibration=model.get('full_stop_calibration'),
        initial_pour_calibration=model.get('initial_pour_calibration'),
        late_drip_alignment=model.get('late_drip_alignment'),
        level_refit=model.get('level_refit'),
        retained_incomplete_attempts=integrity.get('retained_incomplete_attempts',[]),
        container=CID,development=rows,development_round=a.round,development_visual_review_sha256=sha(a.review),
        development_integrity_sha256=sha(STUDY/'development-recording-integrity.json'),
        development_withdrawal_sha256=sha(a.withdrawal),
        development_selection=dict(strict_passes=sum(r['strict_pass'] for r in rows),attempts=8,
            known_full_stop_limitations=a.evaluate_known_full_stop_limitations,
            scope='Explicit benchmark selection may retain documented full-stop failures. All eight must still grasp, transfer, return and have zero measured loss. No outcome is relabeled; reserved scoring is unchanged.'),
        selection_helper_sha256=sha(__file__),selection_gate_sha256=sha(pathlib.Path(__file__).with_name('wide_release_gate.py')),
        primary_definition='Verified grasp and transfer; correct full/empty stop; >=90% settled rim height or empty source; ordered return/release; zero measured final/near-table particle loss.',
        volume_units='Nominal particle-count mL, not calibrated physical volume',
        test_policy='Thirty reserved once-only scenes. No retries, fitting, threshold/source changes or teacher fallback during evaluation.')
    # Write under the owned mount using its normal container permissions.
    code="""import hashlib,json,pathlib,shutil,sys
s=json.load(sys.stdin);r=pathlib.Path('/workspace/learning/A-finish-20261106-v1');src=pathlib.Path('/workspace/src')
assert not (r/'final-selection.json').exists()
assert hashlib.sha256((r/'best.pt').read_bytes()).hexdigest()==s['selected_checkpoint_sha256']
archive=r/'frozen-candidate';archive.mkdir(exist_ok=False);(archive/'src').mkdir()
for n,h in s['runtime_sources'].items():
 assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h;shutil.copy2(src/n,archive/'src'/n)
for n in ('best.pt','model-summary.json','frozen-plan.json','data-manifest.json','calibration-plan.json','calibration-seed-provenance.json','calibration-particle-seed.npz','calibration-anchor-manifest.json'):shutil.copy2(r/n,archive/n)
for n in ('infrastructure-aborts.json','development-resume-plan.json','development-resume-plan-r0.json','recording-repair.json','full-stop-calibration.json','initial-pour-calibration.json','coarse-brake-window-probe.json','late-drip-calibration.json','late-drip-clearance-r3.json','late-loss-diagnostic-r2.json','full-stop-diagnostic-r2.json','level-refit.json','settled-target-supervision.json','runtime-source-manifest-drip-r3.json'):
 if (r/n).exists():shutil.copy2(r/n,archive/n)
payload=json.dumps(s,indent=2)+'\\n';(archive/'final-selection.json').write_text(payload)
p=r/'final-selection.partial.json';p.write_text(payload);p.replace(r/'final-selection.json')
"""
    subprocess.run(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',code],input=json.dumps(selection),text=True,check=True,timeout=90)
    destination.mkdir();snapshot=capture(selection)
    (destination/'runtime-before-tests.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    command('finish_test')
    (destination/'queued.json').write_text(json.dumps(dict(queued_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=30,
        checkpoint_sha256=checkpoint,selection_sha256=sha(STUDY/'final-selection.json')),indent=2)+'\n')
    print('FROZEN_AND_QUEUED_30_FRESH_TESTS',flush=True)


if __name__=='__main__':main()
