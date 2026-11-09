"""Finite read-only delivery after a closed native split; no queue or fitting."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];E=PROJECT/'evidence/side-return-A'
STUDY='/home/kadir/aiworker-pouring-4070/learning/A-side-return-20261108-v1'
NATIVE='/workspace/learning/A-side-return-20261108-v1'
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
def remote(args,**kw):return subprocess.run(['ssh','kfuUbuntu',shlex.join(args)],check=True,**kw)
def native(tool,*args):remote(['docker','exec','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','PYTHONPATH='+NATIVE+'/staged:/workspace/src:/workspace',CID,'/isaac-sim/python.sh',NATIVE+'/'+tool,*args],timeout=3600)
def copy(source,target,directory=False):
    assert not target.exists(),str(target)
    subprocess.run(['scp','-q',*(['-r'] if directory else []),'kfuUbuntu:'+source,str(target)],check=True,timeout=1800)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
    stem=f'{a.part}-r{a.round}';assert not (E/(stem+'-outcomes.json')).exists();assert len(a.checkpoint)==64
    count=6 if a.part=='development' else 30;deadline=time.monotonic()+14*3600
    code='''
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_side_cases import ROOT,read,capture
from watch_wide_collection import health
part,stem,checkpoint=sys.argv[1:];m=read(ROOT/'candidate.json');assert m['checkpoint_sha256']==checkpoint
capture(m);progress=read(ROOT/stem/'progress.json');h=health();assert not h.get('error') and not h.get('learning_error')
assert not progress['health'].get('error')
audit=read(ROOT/'recording-audit.json') if (ROOT/'recording-audit.json').exists() else {'records':[],'started_incomplete':['pending']}
expected={s['name'] for s in read(ROOT/'plan.json')[part]}
ready=progress['status']=='complete' and not h['queue'] and not h['recording']['active'] and not audit['started_incomplete'] and expected<={r['name'] for r in audit['records']}
print(json.dumps(dict(ready=ready,completed=progress['completed'],strict_passes=progress['strict_passes'])))
'''
    previous=None
    while time.monotonic()<deadline:
        result=remote(['python3','-',a.part,stem,a.checkpoint],input=code,text=True,capture_output=True,timeout=40)
        status=json.loads(result.stdout)
        if status!=previous:print(json.dumps(status),flush=True);previous=status
        if status['ready']:break
        time.sleep(30)
    else:raise TimeoutError('Finite delivery wait expired; native work untouched')
    native('summarize_side_tasks.py','--part',a.part,'--require-complete')
    native('review_side_tasks.py','--part',a.part)
    for name,suffix in ((a.part+'-outcomes.json','-outcomes.json'),(a.part+'-visual-review.json','-visual-review.json'),('recording-audit.json','-audit.json')):
        copy(STUDY+'/'+name,E/(stem+suffix))
    for name in ('progress','runtime-before','runtime-after'):copy(STUDY+'/'+stem+'/'+name+'.json',E/(stem+'-'+name+'.json'))
    copy(STUDY+'/'+a.part+'-review',E/(stem+'-native-review'),True)
    outcomes=json.loads((E/(stem+'-outcomes.json')).read_text());assert outcomes['counts']['completed']==count
    env=dict(os.environ,PYTHONPATH=str(PROJECT/'runtime/src')+':'+str(PROJECT/'tools'),PYTHONPYCACHEPREFIX='/tmp/aiworker-side-pycache')
    def local(tool,*args):subprocess.run([sys.executable,str(PROJECT/'tools'/tool),*args],cwd=PROJECT,env=env,check=True,timeout=7200)
    local('backup_side_metadata.py','--phase',stem)
    for batch in ((0,) if a.part=='development' else (1,2,3)):
        for tool in ('render_side_tasks.py','review_side_video.py'):
            local(tool,'--part',a.part,'--round',str(a.round),'--batch',str(batch))
    (E/(stem+'-delivery.json')).write_text(json.dumps(dict(checkpoint_sha256=a.checkpoint,counts=outcomes['counts'],manual_review='pending',helper_sha256=sha(Path(__file__)),scope='All declared attempts audited, scalar-backed-up, rendered and decode-checked. Actual selected-frame manual inspection remains required.'),indent=2)+'\n')
    print('LABELED_VIDEOS_READY_MANUAL_REVIEW_PENDING',flush=True)
if __name__=='__main__':main()
