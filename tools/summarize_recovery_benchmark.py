"""Reproducible local summary and backup verification for the recovery study."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence/recovery-A'


def read(path):return json.loads(path.read_text())


def main():
    result=read(EVIDENCE/'recovery-results.json')
    model=read(EVIDENCE/'model-summary-final.json')
    selection=read(EVIDENCE/'model-selection.json')
    assert result['test_status']=='complete'
    assert result['checkpoint_sha256']==model['checkpoint_sha256']==selection['selected_sha256']
    for name,key in [('best-bc-v6.pt','checkpoint_sha256'),('frozen-plan.json','scene_plan_sha256'),('dataset-manifest.json','dataset_manifest_sha256')]:
        assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest()==model[key],name
    for name,key in [('pour_learning.py','actor_source_sha256'),('pour_recovery_learning.py','policy_source_sha256')]:
        assert hashlib.sha256((ROOT/'runtime/src'/name).read_bytes()).hexdigest()==model[key],name
    tests=result['groups']['autonomous_test'];rows=[]
    expected={'BC6_'+s['name'] for s in read(EVIDENCE/'frozen-plan.json')['test']}
    assert tests['attempts']==6 and {r['name'] for r in tests['rows']}==expected
    for r in tests['rows']:
        assert not r['teacher_fallback'] and r['checkpoint_sha256']==model['checkpoint_sha256']
        local=ROOT/'data/material-A-recovery'/Path(r['path']).parts[-2]/Path(r['path']).name/'episode.hdf5'
        assert local.exists(),str(local)
        rows.append(dict(r,local_recording_path=str(local)))
    summary=dict(status='complete',evidence_kind='native Isaac Sim expert recovery and camera-only policy evaluation',
        model=model,selection=selection,
        expert_recoveries=dict(attempts=8,zero_spill_successes=4,admitted_training=3,admitted_validation=1),
        autonomous_test={k:v for k,v in tests.items() if k!='rows'},
        final_outside_mean_ml=sum(r['final_outside_ml'] for r in rows)/6,
        final_outside_max_ml=max(r['final_outside_ml'] for r in rows),
        definition_note='A stop-class prediction is not a completed stop/return/release. Zero spill alone after no motion is not task success. Volumes are nominal particle volume.',rows=rows)
    (EVIDENCE/'final-benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (EVIDENCE/'final-benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows(dict(r,gates='; '.join(r['gates'])) for r in rows)
    integrity=read(EVIDENCE/'recording-integrity.json');files=[]
    for r in integrity['files']:
        p=ROOT/'data/material-A-recovery'/r['relative_path']
        assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
        files.append(dict(r,local_path=str(p)))
    index=dict(episodes=len(files),all_native_checksums_match=True,
        valid_recordings=integrity['valid_recordings'],total_hdf5_bytes=sum(r['bytes'] for r in files),
        partial_aborted_excluded=integrity['partial_aborted_excluded'],files=files)
    (ROOT/'data/material-A-recovery/backup-index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(expert_recoveries=summary['expert_recoveries'],autonomous_test=summary['autonomous_test'],
        final_outside_mean_ml=summary['final_outside_mean_ml'],verified_native_backups=len(files)),indent=2))


if __name__=='__main__':main()
