"""Audit-only closure after an observer failure; never queues or reruns work."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_side_cases import ROOT,read,sha,capture,w,completed_row,strict_success,reasons
from side_recording_ready import committed_recording_paths
p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert a.output.parent==ROOT and a.output.exists()
m=read(ROOT/'candidate.json');before=capture(m);h=w.health()
assert not h['queue'] and not h['recording']['active'] and h['phase'] in ('Complete','Trial finished with failed checks')
assert not h.get('error') and not h.get('learning_error')
old=read(a.output/'progress.json');assert old['status']=='running' and not (a.output/'runtime-after.json').exists()
prior=read(a.output/'runtime-before.json')
for key in ('checkpoint_sha256','plan_sha256','runtime_sources'):assert prior[key]==m[key]
plan=read(ROOT/'plan.json');expected={s['name']:s for s in plan[a.part]};rows=[];seen=set()
for path in (w.HOST_ROOT/'outputs/fountain').glob('*/*SR_A_*/parameters.json'):
    spec=read(path);name=spec['name']
    if name not in expected:continue
    assert name not in seen;seen.add(name);assert all(spec[k]==v for k,v in expected[name].items())
    native=str(Path('/workspace')/path.parent.relative_to(w.HOST_ROOT))
    assert native in committed_recording_paths(path.parent.parent)
    row=completed_row(path.parent,m['checkpoint_sha256']);row.update(strict_pass=strict_success(row),failure_reasons=reasons(row));rows.append(row)
assert seen==expected.keys() and len(rows)==old['planned']
subprocess.run(['docker','exec',w.CID,'/isaac-sim/python.sh','/workspace/learning/A-side-return-20261108-v1/check_side_stage.py'],check=True)
audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete']
after=capture(m)
archive=a.output/'progress-before-audit-recovery.json';assert not archive.exists();archive.write_bytes((a.output/'progress.json').read_bytes())
(a.output/'runtime-after.json').write_text(json.dumps(after,indent=2)+'\n')
progress=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),part=a.part,status='complete',planned=len(rows),completed=len(rows),strict_passes=sum(r['strict_pass'] for r in rows),cases=rows,health={k:h.get(k) for k in ('phase','sim_time','recording','error')},audit_only_recovery=True)
temp=a.output/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(a.output/'progress.json')
(a.output/'observer-recovery.json').write_text(json.dumps(dict(reason='Original observer saw metrics.json before samples.json was committed. All declared attempts subsequently closed normally.',no_replay_or_queue=True,previous_progress_sha256=sha(archive),audit_sha256=sha(ROOT/'recording-audit.json'),before=before,after=after,helper_sha256=sha(Path(__file__))),indent=2)+'\n')
print(json.dumps(dict(audit_only_recovery=True,completed=len(rows),strict_passes=progress['strict_passes'])))
