"""Finite guarded observer; once-only calibration, never automatic restarts."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, '/tmp')
import watch_wide_collection as w
from side_recording_ready import committed_recording_paths
ROOT = w.HOST_ROOT / 'learning/A-scale-20261109-v1'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, d):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(d, indent=2)+'\n')
    temp.replace(p)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--part', choices=('pilot', 'collect', 'holdout'), required=True)
    a.add_argument('--round', choices=('r0','r1'), default='r0')
    a = a.parse_args()
    output = ROOT / (a.part + '-' + a.round)
    assert not output.exists()
    m = read(ROOT/'candidate.json')
    def check():
        witness = w.isolation()
        assert sha(ROOT/'plan.json') == m['plan_sha256'] and sha(ROOT/'best.pt') == m['checkpoint_sha256']
        assert all(sha(w.HOST_ROOT/'src'/n) == h for n,h in m['runtime_sources'].items())
        h = w.health()
        assert not h.get('error') and not h.get('learning_error')
        return h, witness
    h, witness = check()
    assert not h['queue'] and not (h.get('recording') or {}).get('active')
    plan = read(ROOT/'plan.json'); cases = {s['name']:s for s in plan[a.part]}
    if a.part=='pilot':assert plan.get('pilot_round','r0')==a.round
    assert cases
    preflight = read(ROOT/(a.part+'-geometry.json'))
    assert preflight['complete'] and preflight['passed'] and preflight['plan_sha256'] == m['plan_sha256']
    outputs = w.HOST_ROOT/'outputs/fountain'
    existing = {read(p)['name'] for p in outputs.glob('*/*SC_A_*/parameters.json')}
    assert not cases.keys() & existing, 'Started IDs cannot be retried'
    output.mkdir(); write(output/'runtime-before.json', witness)
    w.command('scale_'+a.part)
    deadline = time.monotonic()+6*3600
    seen = set()
    while time.monotonic() < deadline:
        h, witness = check()
        closed = {}
        for p in outputs.glob('*/*SC_A_*/metrics.json'):
            spec = read(p.parent/'parameters.json'); name = spec['name']
            if name not in cases: continue
            native = str(Path('/workspace')/p.parent.relative_to(w.HOST_ROOT))
            if native not in committed_recording_paths(p.parent.parent): continue
            assert all(spec[k] == v for k,v in cases[name].items())
            samples = read(p.parent/'samples.json'); final = samples[-1]
            closed[name] = dict(name=name,path=native,final_height=final['cup_fill_fraction'],
                                final_outside_ml=final['spilled_ml'],ever_table_ml=final['ever_table_ml'])
            if name not in seen: print(json.dumps(closed[name]),flush=True);seen.add(name)
        write(output/'progress.json',dict(completed=len(closed),planned=len(cases),cases=list(closed.values()),
                                         phase=h['phase'],queue=h['queue'],runtime=witness))
        idle = not h['queue'] and not (h.get('recording') or {}).get('active')
        if len(closed)==len(cases) and idle:
            write(output/'runtime-after.json',witness)
            print('CALIBRATION_SPLIT_COMPLETE',flush=True)
            return
        if idle and closed:
            started = {read(p)['name'] for p in outputs.glob('*/*SC_A_*/parameters.json')} & cases.keys()
            assert started == closed.keys(), 'Incomplete records retained, never retried'
            w.command('scale_'+a.part)
        time.sleep(15)
    raise TimeoutError('Finite calibration observer ended; native state preserved')


if __name__ == '__main__': main()
