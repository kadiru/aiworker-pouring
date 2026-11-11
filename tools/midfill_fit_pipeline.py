"""Sequential admitted-data fit. Never starts reserved tests."""
import json,subprocess,sys
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
def main():
    for name in ('finalize_midfill_review.py','cache_midfill_levels.py','prepare_midfill_levels.py','fit_midfill_levels.py'):
        print(json.dumps(dict(starting=name)),flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/name)],check=True)
    protected_unchanged()
    result=dict(passed=read(ROOT/'pair-r10.json')['passed'],pair_report_sha256=sha(ROOT/'pair-r10.json'),reserved_started=False)
    with (ROOT/'midfill-development.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
