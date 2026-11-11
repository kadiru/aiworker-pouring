"""Read-only final audit of preserved inputs, actual records, and test usage."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
HOST=Path('/home/kadir/aiworker-pouring-4070')
ROOT=HOST/'learning/A-scale-dynamic-20261109-v1'
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def native(p):
    path=Path(p)
    return HOST/path.relative_to('/workspace') if path.is_relative_to('/workspace') else path
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r10','r11','r12','r13','r14','r15','r16','r17'),default='r10');args=parser.parse_args();prefix='pair-'+args.round
    assert not (ROOT/'expanded-repair-closure.json').exists()
    candidate=read(ROOT/'candidate.json');protocol=read(ROOT/'protocol.json')
    assert all(sha(native(n))==h for n,h in candidate['protected_parent_files'].items())
    assert all(sha(HOST/'src'/n)==h for n,h in candidate['runtime_sources'].items())
    old=read(ROOT/'protocol-v5-before-midfill.json')
    assert all(protocol[k]==old[k] for k in ('validation','holdout','gates'))
    retained=read(ROOT/'expanded-retained-inputs.json')
    assert all(sha(Path(v['retained_path']))==v['sha256'] for files in retained.values() for v in files.values())
    for previous in ('r10','r11','r12','r13','r14','r16'):
        if (ROOT/f'pair-{previous}.json').exists():
            assert all(sha(ROOT/n)==h for n,h in read(ROOT/f'pair-{previous}-protocol.json')['frozen_files'].items())
    if (ROOT/'conditioned-r15-interruption.json').exists():
        interruption=read(ROOT/'conditioned-r15-interruption.json')
        assert sha(ROOT/'pair-r15-rgb-preserved-partial.pt')==interruption['partial_checkpoint_sha256']
        assert sha(ROOT/'pair-r15-protocol.json')==interruption['protocol_sha256'] and not (ROOT/'pair-r15.json').exists()
        assert all(sha(ROOT/n)==h for n,h in read(ROOT/'pair-r15-protocol.json')['frozen_files'].items())
    audit=read(ROOT/'train-r3-audit.json');review=read(ROOT/'train-r3-manual-review.json')
    assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ROOT/'train-r3-audit.json')
    assert len(audit['records'])==36 and {r['name'] for r in audit['records']}==set(protocol['midfill_train_names'])
    for r in audit['records']:
        path=native(r['path']);assert sha(path)==r['sha256']
        assert r['label']['accepted'] and r['recording_validation']['valid'] and all(v==0 for v in r['containment'].values())
        assert sha(path.parent/'trajectory.npz')==r['trajectory_sha256']
        assert all(sha(path.parent/n)==h for n,h in r['sidecars'].items())
    for batch in (1,2,3):
        manifest=ROOT/f'train-r3-early-review-b{batch}.json';m=read(manifest);r=read(ROOT/f'train-r3-review-b{batch}.json')
        assert r['accepted'] and r['manifest_sha256']==sha(manifest) and r['selected_camera_frames']==48
        assert all(sha(native(page['path']))==page['sha256'] for page in m['pages'])
    pair=read(ROOT/(prefix+'.json'));cfg=read(ROOT/(prefix+'-protocol.json'))
    assert all(sha(ROOT/n)==h for n,h in cfg['frozen_files'].items())
    for branch in ('rgb','color'):
        r=read(ROOT/f'{prefix}-{branch}.json');assert r['checkpoint_sha256']==sha(ROOT/f'{prefix}-{branch}.pt')
    holdouts={s['name'] for s in protocol['holdout']}
    started={read(p)['name']:p for p in (HOST/'outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json')}
    selection=ROOT/'component-selection.json';evaluation=ROOT/'moving-holdout-evaluation.json'
    if pair['passed']:
        assert selection.exists() and evaluation.exists() and set(started)==holdouts
        frozen=read(selection);result=read(evaluation)
        assert frozen['all_required_development_gates_passed'] and result['selection_sha256']==sha(selection)
        assert all(sha(ROOT/n)==h for n,h in frozen['frozen_files'].items())
        assert len(result['cases'])==18
        reserved=dict(started=18,passed_cases=result['passed_cases'],passed=result['passed'],evaluation_sha256=sha(evaluation))
    else:
        assert not started and not selection.exists() and not evaluation.exists()
        reserved=dict(started=0,passed_cases=None,passed=None,reason='Required development accuracy did not pass')
    storage=json.loads(subprocess.check_output(['python3',str(HOST/'tools/verify_robot_training_storage.py'),'--minimum-free-gib','57'],text=True))
    container=json.loads(subprocess.check_output(['docker','inspect',CID],text=True))[0]
    caps=container['HostConfig'];assert caps['DeviceRequests'][0]['DeviceIDs']==['GPU-cde23916-c5fa-f18e-a343-127f732cc2f6']
    assert caps['NanoCpus']==4000000000 and caps['CpusetCpus']=='16-23' and caps['Memory']==caps['MemorySwap']==17179869184
    tracker_name='tracker-t3' if args.round in ('r12','r13','r14','r15','r16','r17') else 'tracker-t2'
    report=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tracker=tracker_name,tracker_passed=read(ROOT/(tracker_name+'.json'))['passed'],
        new_tracker_admission_passed=read(ROOT/'midfill-tracker-admission.json')['passed'],round=args.round,pair_passed=pair['passed'],
        reserved=reserved,new_training_records_verified=36,selected_camera_frames_reviewed=144,
        parent_files_verified=len(candidate['protected_parent_files']),runtime_sources_verified=len(candidate['runtime_sources']),
        active_parent_checkpoint_sha256=sha(ROOT/'best.pt'),protocol_sha256=sha(ROOT/'protocol.json'),
        validation_reserved_and_gates_unchanged=True,storage=storage,simulator_running=container['State']['Running'],
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip(),
        helper_sha256=sha(Path(__file__)),autonomy_accepted=False,
        scope='Known-size simulated perception repair; no autonomous pouring claim. No foreign job signaled or reconfigured. Retained failures and unused/once-only reserved status verified explicitly.')
    with (ROOT/'expanded-repair-closure.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
