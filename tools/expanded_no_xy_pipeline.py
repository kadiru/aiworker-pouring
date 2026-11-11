"""Sequential development ablation; never consumes a reserved fixture."""
import json,subprocess,sys
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def main():
    for name in ('prepare_expanded_levels_no_xy.py','fit_expanded_levels_no_xy.py'):
        subprocess.run([sys.executable,'-u',str(ROOT/name)],check=True)
    protected_unchanged()
    result=dict(passed=read(ROOT/'pair-r9.json')['passed'],tracker_passed=read(ROOT/'tracker-t2.json')['passed'],
                pair_report_sha256=sha(ROOT/'pair-r9.json'),reserved_started=False)
    with (ROOT/'expanded-no-xy-development.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
