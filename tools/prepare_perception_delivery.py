"""Finite read-only handoff from an audited six-case split to video drafts.

Never fits, queues, retries, restarts, changes scoring, or marks manual review.
The existing native observer owns simulation/source/resource monitoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / 'evidence/perception-A'
STUDY = '/home/kadir/aiworker-pouring-4070/learning/A-perception-20261108-v1'
NATIVE = '/workspace/learning/A-perception-20261108-v1'
CID = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'

POLL = r'''
import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_wide_collection import health,CID
part,round_number,checkpoint=sys.argv[1:]
root=Path('/home/kadir/aiworker-pouring-4070/learning/A-perception-20261108-v1')
read=lambda p:json.loads(p.read_text())
assert subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',CID],text=True).strip()=='true','Priority stop respected; no restart'
manifest=read(root/'candidate.json')
assert manifest['checkpoint_sha256']==checkpoint and manifest['perception_round']==int(round_number)
assert hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()==checkpoint
plan=read(root/'plan.json');expected={s['name'] for s in plan[part]};assert len(expected)==6
progress=read(Path('/tmp/aiworker-perception-'+part+'-r'+round_number)/'progress.json')
assert progress['part']==part and not progress['health'].get('error')
assert progress['status'] in ('running','complete')
h=health();assert not h.get('error') and not h.get('learning_error')
audit=read(root/'recording-audit.json')
ready=(progress['status']=='complete' and progress['completed']==6
       and not h['queue'] and not h['recording']['active']
       and not audit['started_incomplete']
       and expected<={r['name'] for r in audit['records']})
print(json.dumps(dict(ready=ready,completed=progress['completed'],strict_passes=progress['strict_passes'])))
'''


def remote(arguments, **kwargs):
    return subprocess.run(['ssh', 'kfuUbuntu', shlex.join(arguments)],
                          check=True, **kwargs)


def native(tool, *arguments):
    remote(['docker', 'exec', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=2',
            CID, '/isaac-sim/python.sh', NATIVE+'/'+tool, *arguments], timeout=1800)


def copy(source, target, directory=False):
    assert not target.exists(), f'Preserve existing artifact: {target}'
    subprocess.run(['scp', '-q', *(['-r'] if directory else []),
                    'kfuUbuntu:'+source, str(target)], check=True, timeout=1800)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=('development', 'test'), required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    assert args.round >= 0
    assert len(args.checkpoint) == 64 and all(c in '0123456789abcdef' for c in args.checkpoint)
    stem = f'{args.part}-r{args.round}'
    assert not (EVIDENCE/(stem+'-outcomes.json')).exists()
    selection = json.loads((EVIDENCE/f'component-selection-r{args.round}.json').read_text())
    assert selection['checkpoint_sha256'] == args.checkpoint
    deadline = time.monotonic()+6*3600
    previous = None
    while time.monotonic() < deadline:
        result = remote(['python3', '-', args.part, str(args.round), args.checkpoint],
                        input=POLL, text=True, capture_output=True, timeout=30)
        status = json.loads(result.stdout)
        if status != previous:
            print(json.dumps(status), flush=True)
            previous = status
        if status['ready']:
            break
        time.sleep(30)
    else:
        raise TimeoutError('Finite delivery wait expired; native work unchanged')
    print('SIX_RECORDINGS_AUDITED_PREPARING_DELIVERY', flush=True)
    native('summarize_perception_tasks.py', '--part', args.part, '--require-complete')
    native('review_perception_tasks.py', '--part', args.part)
    native('analyze_perception_particle_motion.py', '--part', args.part, '--round', str(args.round))
    for source, suffix in ((args.part+'-outcomes.json', '-outcomes.json'),
                           (args.part+'-visual-review.json', '-visual-review.json'),
                           ('recording-audit.json', '-audit.json'),
                           (f'{args.part}-particle-motion-r{args.round}.json', '-particle-motion.json')):
        copy(STUDY+'/'+source, EVIDENCE/(stem+suffix))
    snapshots = '/tmp/aiworker-perception-'+stem
    for name in ('progress', 'runtime-before', 'runtime-after'):
        copy(snapshots+'/'+name+'.json', EVIDENCE/(stem+'-'+name+'.json'))
    copy(STUDY+'/'+args.part+'-review', EVIDENCE/(stem+'-native-review'), directory=True)
    outcomes = json.loads((EVIDENCE/(stem+'-outcomes.json')).read_text())
    assert outcomes['checkpoint_sha256'] == args.checkpoint and outcomes['counts']['completed'] == 6
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(PROJECT/'runtime/src')+os.pathsep+str(PROJECT/'tools')
    for tool, arguments in (
        ('backup_perception_metadata.py', ['--phase', stem]),
        ('render_perception_tasks.py', ['--part', args.part, '--round', str(args.round)]),
        ('review_perception_video.py', ['--part', args.part, '--round', str(args.round)]),
    ):
        subprocess.run([sys.executable, str(PROJECT/'tools'/tool), *arguments],
                       cwd=PROJECT, env=environment, check=True, timeout=3600)
    report = dict(checkpoint_sha256=args.checkpoint, counts=outcomes['counts'],
                  visual_review='pending', helper_sha256=sha(Path(__file__)),
                  artifacts={name:sha(EVIDENCE/(stem+name)) for name in (
                      '-outcomes.json', '-audit.json', '-visual-review.json',
                      '-particle-motion.json', '-video.json', '-video-review.json')},
                  scope='Six closed, audited native attempts backed up and rendered. All outcomes retained. Selected native and encoded frames still require actual manual inspection; no fitting, test launch or candidate freeze.')
    (EVIDENCE/(stem+'-delivery.json')).write_text(json.dumps(report, indent=2)+'\n')
    print('VIDEOS_READY_MANUAL_REVIEW_PENDING', flush=True)


if __name__ == '__main__':
    main()
