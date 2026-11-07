"""Finite read-only observer for one development round or frozen 30-test run."""
import argparse,datetime,hashlib,json,pathlib,subprocess,sys,time
from watch_wide_collection import isolation,health,CID,STUDY,HOST_ROOT

STUDY=HOST_ROOT/'learning/A-finish-20261106-v1'
sys.path.insert(0,str(HOST_ROOT/'src'))
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def capture(metadata):
    witness=isolation();sources=metadata.get('runtime_sources',metadata.get('sources'))
    checkpoint=metadata.get('selected_checkpoint_sha256',metadata.get('checkpoint_sha256'))
    assert digest(STUDY/'best.pt')==checkpoint
    for name,sha in sources.items():assert digest(HOST_ROOT/'src'/name)==sha,name
    def out(*args):return subprocess.check_output(args,text=True,timeout=20).strip()
    return dict(checked_utc=witness['utc'],container=CID,limits=witness['limits'],guard_processes=witness['guard'].splitlines(),
        selected_checkpoint_sha256=checkpoint,sources=sources,
        asset_sha256=digest(HOST_ROOT/'assets/FFW_BG2_without_mimic.usd'),
        gpu_compute_processes=witness['gpu_processes'],
        gpu_utilization=out('nvidia-smi','--query-gpu=name,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'),
        memory=out('free','-m'),
        health={k:v for k,v in health().items() if k in ('run','phase','sim_time','queue','recording','error','learning_error','learning')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('development','test'),required=True)
    parser.add_argument('--round',type=int,default=0);parser.add_argument('--output-dir',required=True);a=parser.parse_args()
    destination=pathlib.Path(a.output_dir);destination.mkdir(exist_ok=False)
    metadata=read(STUDY/('final-selection.json' if a.part=='test' else 'model-summary.json'))
    sources=metadata.get('runtime_sources',metadata.get('sources'));checkpoint=metadata.get('selected_checkpoint_sha256',metadata.get('checkpoint_sha256'))
    plan=read(STUDY/'frozen-plan.json');names={s['name']+(f'_r{a.round}' if a.part=='development' else '') for s in plan[a.part]}
    assert len(names)==(30 if a.part=='test' else 8)
    seen={};started=time.monotonic();snapshots=set()
    def snapshot(label):
        record=capture(metadata);record['completed_test_cases_at_observation']=sorted(seen)
        (destination/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n');snapshots.add(label)
    snapshot('runtime-before-observation')
    while time.monotonic()-started<12*3600:
        assert digest(STUDY/'best.pt')==checkpoint
        for n,sha in sources.items():assert digest(HOST_ROOT/'src'/n)==sha,n
        if a.part=='test':assert digest(STUDY/'frozen-plan.json')==metadata['scene_plan_sha256']
        for p in sorted((HOST_ROOT/'outputs/fountain').glob('*/*FN30_A_*/metrics.json')):
            spec=read(p.parent/'parameters.json');name=spec['name']
            if name not in names or name in seen:continue
            assert spec['protocol']==18
            row=completed_row(p.parent,checkpoint);row.update(strict_pass=strict_success(row),failure_reasons=reasons(row))
            seen[name]=row
            print(json.dumps({k:row[k] for k in ('name','strict_pass','failure_reasons','stop_reason','cup_surface_height_fraction','source_remaining_ml','final_outside_ml')}),flush=True)
        h=health();assert not h.get('error') and not h.get('learning_error'),h.get('error') or h.get('learning_error')
        progress=dict(stage='complete' if set(seen)==names else 'running',part=a.part,round=a.round,
            completed=len(seen),planned=len(names),strict_passes=sum(r['strict_pass'] for r in seen.values()),
            rows=list(seen.values()),health={k:h.get(k) for k in ('phase','sim_time','recording','error')})
        temp=destination/'progress.partial.json';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(destination/'progress.json')
        for n in ((5,15,25) if a.part=='test' else (4,)):
            label=f'runtime-after-test{n:02}'
            if len(seen)>=n and label not in snapshots:snapshot(label)
        if set(seen)==names:
            assert not h['queue'] and not h.get('recording',{}).get('active')
            snapshot('runtime-final');print('ALL_CASES_COMPLETE',flush=True);return
        time.sleep(30)
    raise TimeoutError('Finite observation limit expired; all native artifacts remain intact')


if __name__=='__main__':main()
