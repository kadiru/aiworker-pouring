"""Finite sequential validation and fit; never starts reserved collection."""
import json
import subprocess
import sys
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def run(name):
    print(json.dumps(dict(starting=name)),flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/name)],check=True)

def main():
    assert not (ROOT/'expanded-perception-development.json').exists()
    for name in ('check_expanded_perception.py','prepare_expanded_tracker.py','evaluate_expanded_tracker.py'):
        run(name)
    tracker=read(ROOT/'tracker-t2.json');pair=None
    if tracker['passed']:
        for name in ('cache_expanded_levels.py','prepare_expanded_levels.py','fit_expanded_levels.py'):
            run(name)
        pair=read(ROOT/'pair-r8.json')
    protected_unchanged()
    result=dict(passed=bool(tracker['passed'] and pair and pair['passed']),tracker_passed=tracker['passed'],
                level_pair_passed=None if pair is None else pair['passed'],reserved_started=False,
                tracker_report_sha256=sha(ROOT/'tracker-t2.json'),
                pair_report_sha256=sha(ROOT/'pair-r8.json') if pair else None,
                scope='Development-only repair. Reserved evaluation requires a separate verified freeze and launch.')
    with (ROOT/'expanded-perception-development.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
