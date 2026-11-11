"""Verify geometric reconstruction and fit separate models; never start holdouts."""
import json
import os
import subprocess
import sys
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    assert not read(ROOT/'pair-r13.json')['passed']
    result=subprocess.run([sys.executable,'-m','unittest','-v','test_landmark_level_geometry'],cwd=ROOT,
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    checks=dict(passed=result.returncode==0,output=result.stdout+result.stderr,
                source_sha256=sha(ROOT/'landmark_level_model.py'),test_sha256=sha(ROOT/'test_landmark_level_geometry.py'))
    with (ROOT/'landmark-geometry-contracts.json').open('x') as f:
        json.dump(checks,f,indent=2);f.write('\n')
    print(json.dumps(checks),flush=True);assert checks['passed']
    for name in ('prepare_landmark_level_data.py','fit_landmark_levels.py'):
        print(json.dumps(dict(starting=name)),flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/name)],check=True)
    protected_unchanged()
    report=dict(passed=read(ROOT/'pair-r14.json')['passed'],pair_report_sha256=sha(ROOT/'pair-r14.json'),reserved_started=False)
    with (ROOT/'landmark-level-development.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
