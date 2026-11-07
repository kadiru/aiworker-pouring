"""Finite handoff from a closed development round to verified local review.

Never trains, queues a test, restarts a container, or changes a recording.
The native observer remains responsible for live source/resource checks.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
STUDY = '/home/kadir/aiworker-pouring-4070/learning/A-finish-20261106-v1'
CID = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    assert args.round >= 0
    assert len(args.checkpoint) == 64 and all(c in '0123456789abcdef' for c in args.checkpoint)
    evidence = PROJECT / 'evidence/finish-A'
    output = evidence / f'development-r{args.round}-visual'
    assert not output.exists(), 'Preserve prior review artifacts'
    remote_progress = f'/tmp/aiworker-finish-development-r{args.round}/progress.json'
    started = time.monotonic()
    while time.monotonic() - started < 3 * 3600:
        progress = json.loads(subprocess.check_output(
            ['ssh', 'kfuUbuntu', 'cat ' + remote_progress], text=True, timeout=25))
        assert progress['part'] == 'development' and progress['round'] == args.round
        assert not progress['health'].get('error')
        if progress['stage'] == 'complete':
            break
        assert progress['stage'] == 'running'
        time.sleep(30)
    else:
        raise TimeoutError('Finite development wait expired; native work is unchanged')
    assert len(progress['rows']) == progress['completed'] == 8
    assert all(r['checkpoint_sha256'] == args.checkpoint for r in progress['rows'])
    print('DEVELOPMENT_CLOSED_STARTING_AUDIT', flush=True)
    code = """import sys,hashlib,pathlib
sys.path.insert(0,'/tmp')
from watch_wide_collection import isolation,health
STUDY=pathlib.Path("/home/kadir/aiworker-pouring-4070/learning/A-finish-20261106-v1")
isolation();h=health()
assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
assert not (STUDY/'final-selection.json').exists()
assert hashlib.sha256((STUDY/'best.pt').read_bytes()).hexdigest()==sys.argv[1]
"""
    subprocess.run(['ssh', 'kfuUbuntu', 'python3 - ' + args.checkpoint],
                   input=code, text=True, check=True, timeout=30)
    command = ('docker exec ' + CID + ' /isaac-sim/python.sh '
               '/workspace/src/audit_finish_study.py '
               '--output development-recording-integrity.json')
    subprocess.run(['ssh', 'kfuUbuntu', command], check=True, timeout=1200)
    integrity = evidence / f'development-recording-integrity-r{args.round}.json'
    local_progress = evidence / f'development-r{args.round}-progress.json'
    for source, target in [(STUDY + '/development-recording-integrity.json', integrity),
                           (remote_progress, local_progress)]:
        subprocess.run(['scp', '-q', 'kfuUbuntu:' + source, str(target)], check=True)
    subprocess.run([sys.executable,str(PROJECT/'tools/backup_finish_metadata.py'),
        '--manifest',str(integrity)],check=True)
    subprocess.run([sys.executable,str(PROJECT/'tools/audit_finish_withdrawal.py'),
        '--manifest',str(integrity),'--round',str(args.round),
        '--output',str(evidence/f'development-r{args.round}-withdrawal.json')],check=True)
    subprocess.run([sys.executable, str(PROJECT / 'tools/review_finish_development.py'),
                    '--manifest', str(integrity), '--progress', str(local_progress),
                    '--output', str(output)], check=True)
    print('NATIVE_AUDIT_AND_BACKUP_COMPLETE_MANUAL_REVIEW_REQUIRED', flush=True)


if __name__ == '__main__':
    main()
