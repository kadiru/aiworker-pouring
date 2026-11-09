"""One isolated 4070 component fit; never activates models or queues tasks."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-20261109-v1'
sys.path.insert(0,str(ROOT))
from scale_data_admission import admit


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--component',choices=('locator','level'),default='locator');args=parser.parse_args()
    helper='fit_scale_locator.py' if args.component=='locator' else 'fit_scale_level.py'
    report_name='cup-neural-localization.json' if args.component=='locator' else 'static-level-fit.json'
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');audit=read(ROOT/'collect-r0-audit.json')
    assert not manifest['trained'] and not (ROOT/'component-selection.json').exists()
    assert audit['complete']
    records=admit(plan,manifest,audit)
    review=read(ROOT/'collect-r0-manual-review.json')
    assert review['accepted'] and review['audit_sha256']==sha(ROOT/'collect-r0-audit.json')
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}, 'Holdout must remain unstarted before model selection'
    assert sha(ROOT/'plan.json')==manifest['plan_sha256'] and sha(ROOT/'best.pt')==manifest['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in manifest['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==d for p,d in manifest['protected_parent_files'].items())
    witness=w.isolation();h=w.health()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error')
    own={int(x.strip()) for x in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in witness['gpu_processes'].splitlines() if r.startswith(w.GPU))
    assert not (ROOT/report_name).exists()
    if args.component=='level':
        assert read(ROOT/'scope-selection.json')['passed']
    output=ROOT/(args.component+'-fit-r0');output.mkdir()
    sources={n:sha(ROOT/n) for n in (helper,'scale_data_admission.py')}
    (output/'runtime-before.json').write_text(json.dumps(dict(witness,sources=sources,admitted_episodes=len(records)),indent=2)+'\n')
    with (output/'fit.log').open('x') as log:
        result=subprocess.run(['docker','exec','-e','PYTHONDONTWRITEBYTECODE=1',w.CID,'nice','-n','10',
            '/isaac-sim/python.sh','/workspace/learning/A-scale-20261109-v1/'+helper],stdout=log,stderr=subprocess.STDOUT)
    (output/'runtime-after.json').write_text(json.dumps(w.isolation(),indent=2)+'\n')
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(ROOT/n)==h for n,h in sources.items())
    report=read(ROOT/report_name) if (ROOT/report_name).exists() else None
    completion=dict(exit_code=result.returncode,passed=None if report is None else report['passed'],
                    model_activated=False,component=args.component,training_method='Supervised component fit; equal size/episode weighting',
                    sources=sources,report_sha256=sha(ROOT/report_name) if report else None)
    (output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion),flush=True)
    assert result.returncode==0


if __name__=='__main__':main()
