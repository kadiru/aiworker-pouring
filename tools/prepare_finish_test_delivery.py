"""Finite read-only handoff from closed tests to audited local video drafts with temporary raw-source downloads.

Does not queue/retry tests, fit, change simulator state, or claim manual review.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / 'evidence/finish-A'
STUDY = '/home/kadir/aiworker-pouring-4070/learning/A-finish-20261106-v1'
CID = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--snapshot-dir', required=True)
    args = parser.parse_args()
    assert len(args.checkpoint) == 64 and all(c in '0123456789abcdef' for c in args.checkpoint)
    assert args.snapshot_dir.startswith('/tmp/aiworker-finish-')
    assert all(c.isalnum() or c in '/_-' for c in args.snapshot_dir)
    assert not (EVIDENCE / 'final-benchmark.json').exists()
    started = time.monotonic()
    while time.monotonic()-started < 12*3600:
        progress = json.loads(subprocess.check_output(
            ['ssh','kfuUbuntu','cat /tmp/aiworker-finish-test/progress.json'],text=True,timeout=25))
        assert progress['part'] == 'test' and not progress['health'].get('error')
        assert all(r['checkpoint_sha256'] == args.checkpoint for r in progress['rows'])
        if progress['stage'] == 'complete':
            break
        assert progress['stage'] == 'running'
        time.sleep(30)
    else:
        raise TimeoutError('Finite test wait expired; native work is unchanged')
    assert len(progress['rows']) == progress['completed'] == 30
    print('THIRTY_TESTS_CLOSED_STARTING_AUDIT',flush=True)
    for command in (
        '/workspace/src/pour_finish_report.py --require-thirty',
        '/workspace/src/audit_finish_study.py --output recording-integrity.json',
    ):
        subprocess.run(['ssh','kfuUbuntu','docker exec '+CID+' /isaac-sim/python.sh '+command],
                       check=True,timeout=1800)
    names = ('finish-results.json','final-benchmark.csv','recording-integrity.json',
             'final-selection.json','frozen-plan.json','data-manifest.json',
             'calibration-plan.json','calibration-seed-provenance.json','calibration-anchor-manifest.json','model-summary.json','best.pt',
             'infrastructure-aborts.json','development-resume-plan.json','recording-repair.json')
    for name in names:
        subprocess.run(['scp','-q','kfuUbuntu:'+STUDY+'/'+name,str(EVIDENCE/name)],check=True)
    for name in ('runtime-after-test05.json','runtime-after-test15.json',
                 'runtime-after-test25.json','runtime-final.json','progress.json'):
        target = 'test-progress.json' if name == 'progress.json' else name
        subprocess.run(['scp','-q','kfuUbuntu:/tmp/aiworker-finish-test/'+name,str(EVIDENCE/target)],check=True)
    subprocess.run(['scp','-q','kfuUbuntu:'+args.snapshot_dir+'/runtime-before-tests.json',
                    str(EVIDENCE/'runtime-before-tests.json')],check=True)
    subprocess.run([sys.executable,str(PROJECT/'tools/backup_finish_metadata.py'),
        '--manifest',str(EVIDENCE/'recording-integrity.json')],check=True)
    subprocess.run([sys.executable,str(PROJECT/'tools/summarize_finish_benchmark.py')],check=True)
    subprocess.run([sys.executable,str(PROJECT/'tools/audit_finish_withdrawal.py'),
        '--manifest',str(EVIDENCE/'recording-integrity.json'),'--output',str(EVIDENCE/'test-withdrawal.json')],check=True)
    for part in (1,2,3):
        subprocess.run([sys.executable,str(PROJECT/'tools/render_finish_videos.py'),'--part',str(part)],check=True)
        subprocess.run([sys.executable,str(PROJECT/'tools/review_finish_videos.py'),'--part',str(part)],check=True)
    print('AUDITED_VIDEOS_READY_MANUAL_REVIEW_REQUIRED',flush=True)


if __name__ == '__main__':
    main()
