"""Test initialization, then fit geometry-conditioned landmarks without holdouts."""
import json,os,subprocess,sys
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged


def main():
    assert not read(ROOT/'pair-r14.json')['passed']
    result=subprocess.run([sys.executable,'-m','unittest','-v','test_conditioned_landmarks'],cwd=ROOT,
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    checks=dict(passed=result.returncode==0,output=result.stdout+result.stderr,
                source_sha256=sha(ROOT/'landmark_level_model_v2.py'),test_sha256=sha(ROOT/'test_conditioned_landmarks.py'))
    with (ROOT/'conditioned-landmark-contracts.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')
    print(json.dumps(checks),flush=True);assert checks['passed']
    for name in ('prepare_conditioned_landmarks.py','fit_conditioned_landmarks.py'):
        print(json.dumps(dict(starting=name)),flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/name)],check=True)
    protected_unchanged();report=dict(passed=read(ROOT/'pair-r15.json')['passed'],pair_report_sha256=sha(ROOT/'pair-r15.json'),reserved_started=False)
    with (ROOT/'conditioned-landmark-development.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
