"""Gated 279-episode localization and level repair; never launches reserved tests."""
import json
import os
import subprocess
import sys
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    assert not read(ROOT/'pair-r11.json')['passed']
    result=subprocess.run([sys.executable,'-m','unittest','-v','test_perception_midfill_data'],cwd=ROOT,
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    checks=dict(passed=result.returncode==0,output=result.stdout+result.stderr,
                source_sha256=sha(ROOT/'perception_midfill_data.py'),test_sha256=sha(ROOT/'test_perception_midfill_data.py'))
    with (ROOT/'midfill-localization-data-contracts.json').open('x') as f:
        json.dump(checks,f,indent=2);f.write('\n')
    print(json.dumps(checks),flush=True);assert checks['passed']
    for name in ('prepare_midfill_rim.py', 'fit_midfill_rim.py', 'prepare_midfill_tracker.py',
                 'evaluate_midfill_tracker.py', 'cache_relocalized_midfill.py',
                 'prepare_relocalized_midfill_levels.py', 'fit_relocalized_midfill_levels.py'):
        print(json.dumps(dict(starting=name)), flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT/name)], check=True)
    protected_unchanged()
    report = dict(passed=read(ROOT/'pair-r12.json')['passed'],
                  tracker_passed=read(ROOT/'tracker-t3.json')['passed'],
                  pair_report_sha256=sha(ROOT/'pair-r12.json'), reserved_started=False)
    with (ROOT/'relocalized-midfill-development.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
