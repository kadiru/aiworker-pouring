"""One guarded GPU revalidation of frozen weights, with no test collection."""
import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-20261109-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    m=read(ROOT/'candidate.json');witness=w.isolation();h=w.health()
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert read(ROOT/m['collection_review'])['accepted'] and read(ROOT/m['collection_audit'])['complete']
    own={int(x.strip()) for x in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in witness['gpu_processes'].splitlines() if r.startswith(w.GPU))
    output=ROOT/'locator-revalidation-r10';output.mkdir();source=sha(ROOT/'revalidate_scale_locator.py');checkpoint=sha(ROOT/'cup-neural-localization.pt')
    (output/'runtime-before.json').write_text(json.dumps(witness,indent=2)+'\n')
    with (output/'check.log').open('x') as f:
        result=subprocess.run(['docker','exec','-e','PYTHONDONTWRITEBYTECODE=1',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-scale-20261109-v1/revalidate_scale_locator.py'],stdout=f,stderr=subprocess.STDOUT)
    assert checkpoint==sha(ROOT/'cup-neural-localization.pt') and source==sha(ROOT/'revalidate_scale_locator.py')
    report=dict(exit_code=result.returncode,runtime=w.isolation(),checkpoint_unchanged=True,source_sha256=source,report_sha256=sha(ROOT/'locator-revalidation-r10.json') if (ROOT/'locator-revalidation-r10.json').exists() else None)
    (output/'completion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);assert result.returncode==0
if __name__=='__main__':main()
