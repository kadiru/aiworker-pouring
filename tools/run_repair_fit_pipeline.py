"""Finite sequential review/admission/fit pipeline; no implicit heldout run."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-dynamic-20261109-v1'

def cpu(helper):
    subprocess.run(['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-scale-dynamic-20261109-v1/'+helper],check=True)

def main():
    w.isolation();h=w.health()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert json.loads((ROOT/'train-r2-pipeline-completion.json').read_text())['completed']
    assert all((ROOT/f'train-r2-review-b{i}.json').exists() for i in (1,2,3))
    cpu('finalize_static_repair_review.py')
    subprocess.run([sys.executable,str(ROOT/'run_dynamic_component.py'),'--helper','cache_static_repair_collection.py','--label','static-repair-cache-v3'],check=True)
    assert json.loads((ROOT/'level-cache-v3.json').read_text())['passed']
    cpu('prepare_level_repair_v3.py')
    subprocess.run([sys.executable,str(ROOT/'run_dynamic_component.py'),'--helper','fit_level_repair_v3.py','--round','r7','--label','pair-r7-fit'],check=True)
    report=dict(fit_completed=True,all_development_gates_passed=json.loads((ROOT/'pair-r7.json').read_text())['passed'],reserved_started=False,runtime=w.isolation())
    with (ROOT/'level-repair-r7-pipeline-completion.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
