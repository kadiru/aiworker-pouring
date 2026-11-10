"""Idle-only bounded component jobs in the immutable owned 4070 container."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-dynamic-20261109-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--helper',choices=['prepare_dynamic_perception.py','fit_dynamic_color.py','fit_dynamic_color_mask.py','fit_dynamic_pair.py','fit_dynamic_rim.py','diagnose_dynamic_rim.py','fit_dynamic_coarse.py'],required=True);p.add_argument('--label',required=True);p.add_argument('--cpu',action='store_true');p.add_argument('--round',choices=('r0','r1','r3','r4','d1','d2'));a=p.parse_args()
    assert a.label.replace('-','').isalnum();before=w.isolation();h=w.health()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    own={int(s.strip()) for s in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in before['gpu_processes'].splitlines() if r.startswith(w.GPU))
    output=ROOT/a.label;output.mkdir();source=sha(ROOT/a.helper)
    (output/'runtime-before.json').write_text(json.dumps(before,indent=2)+'\n')
    cmd=['docker','exec','-e','PYTHONDONTWRITEBYTECODE=1']
    if a.cpu:cmd+=['-e','CUDA_VISIBLE_DEVICES=']
    cmd += [w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-scale-dynamic-20261109-v1/'+a.helper]
    if a.round:
        assert (a.helper=='fit_dynamic_color.py' and a.round in ('r0','r1')) or (a.helper=='fit_dynamic_pair.py' and a.round in ('r3','r4')) or (a.helper=='fit_dynamic_rim.py' and a.round=='d1') or (a.helper=='fit_dynamic_coarse.py' and a.round in ('d1','d2'));cmd+=['--round',a.round]
    with (output/'run.log').open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    assert sha(ROOT/a.helper)==source
    report=dict(exit_code=r.returncode,helper=a.helper,helper_sha256=source,runtime=w.isolation())
    (output/'completion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);assert r.returncode==0
if __name__=='__main__':main()
