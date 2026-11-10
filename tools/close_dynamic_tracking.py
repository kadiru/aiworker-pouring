"""Verify the tracking follow-up without changing runtime or granting test access."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-dynamic-20261109-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=read(ROOT/'candidate.json');h=w.health();witness=w.isolation()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not m['autonomy_accepted'] and not (ROOT/'component-selection.json').exists() and not (ROOT/'autonomy-selection.json').exists()
    assert not (ROOT/'deny-concurrent-probe').exists()
    assert sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/Path(n).relative_to('/workspace'))==v for n,v in m['protected_parent_files'].items())
    assert all(sha(w.HOST_ROOT/'src'/n)==v for n,v in m['runtime_sources'].items())
    assert not list((w.HOST_ROOT/'outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    processes=subprocess.check_output(['docker','top',w.CID,'-eo','pid,args'],text=True)
    assert not any('/workspace/learning/A-scale-dynamic-20261109-v1/' in line and '/isaac-sim' in line for line in processes.splitlines())
    tracker=read(ROOT/'tracker-t1.json');pair=read(ROOT/'pair-r4.json');assert tracker['passed'] and not pair['passed']
    archives=[]
    for prefix,artifacts in (('tracker-t1',['tracker-t1.json','tracker-t1.pt']),('pair-r4',['pair-r4.json','pair-r4-rgb.json','pair-r4-color.json','pair-r4-rgb.pt','pair-r4-color.pt','level-localization-check.json','dynamic-localization-check.json'])):
        protocol=prefix+'-protocol.json';cfg=read(ROOT/protocol);folder=ROOT/(prefix+'-frozen-inputs');folder.mkdir()
        for n,v in cfg['frozen_files'].items():
            source=ROOT/n;assert sha(source)==v;dest=folder/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(dest)==v
        for n in [protocol]+artifacts:shutil.copy2(ROOT/n,folder/n);assert sha(folder/n)==sha(ROOT/n)
        archives.append(dict(path=str(folder),protocol_sha256=sha(ROOT/protocol),frozen_inputs=len(cfg['frozen_files'])))
    reports={n:sha(ROOT/n) for n in ('tracker-t0.json','tracker-t1.json','pair-r4.json','pair-r4-rgb.json','pair-r4-color.json','level-localization-check.json','dynamic-localization-check.json')}
    jobs={name:dict(before=read(ROOT/name/'runtime-before.json'),completion=read(ROOT/name/'completion.json'),log_sha256=sha(ROOT/name/'run.log')) for name in ('tracker-eval-t0','tracker-eval-t1','pair-fit-r4')}
    report=dict(status='tracking_accepted_level_pair_rejected_reserved_unused',runtime=witness,health={k:h.get(k) for k in ('run','phase','queue','recording','error','learning_error')},active_checkpoint_sha256=m['checkpoint_sha256'],protected_parent_files_verified=len(m['protected_parent_files']),runtime_sources_verified=len(m['runtime_sources']),reused_development_episodes=207,new_native_scenes_this_followup=0,reserved_scenes_started=0,autonomy_accepted=False,no_active_controller_change=True,component_jobs_running=False,archives=archives,reports=reports,jobs=jobs,helper_sha256=sha(Path(__file__)),scope='Tracked perception accepted only for this declared known-size stationary-cup scope. RGB/color development gates fail. All 18 reserved scenes remain unused. No foreign job signaled or reconfigured; point-in-time snapshots do not prove uninterrupted execution.')
    with (ROOT/'tracking-closure.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runtime','health','jobs','reports','archives')}),flush=True)
if __name__=='__main__':main()
