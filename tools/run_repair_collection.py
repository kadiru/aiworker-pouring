"""One finite guarded training-data collection followed by a native audit."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-dynamic-20261109-v1'
def main():
    before=w.isolation();h=w.health()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    own={int(s.strip()) for s in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in before['gpu_processes'].splitlines() if r.startswith(w.GPU))
    processes=subprocess.check_output(['docker','top',w.CID,'-eo','pid,args'],text=True)
    assert not any('/workspace/learning/A-scale-dynamic-20261109-v1/' in line and '/isaac-sim' in line for line in processes.splitlines())
    assert json.loads((ROOT/'repair-collection-activation.json').read_text())['extra_training']==36
    subprocess.run([sys.executable,str(ROOT/'watch_repair_collection.py'),'--part','train','--round','r1'],check=True)
    subprocess.run(['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-scale-dynamic-20261109-v1/check_repair_collection.py','--part','train','--round','r1'],check=True)
    with (ROOT/'train-r1-pipeline-completion.json').open('x') as f:json.dump(dict(completed=True,runtime=w.isolation()),f,indent=2);f.write('\n')
    print('REPAIR_COLLECTION_AND_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
