"""Host-only preservation audit after the observed Docker daemon restart."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

HOST = Path('/home/kadir/aiworker-pouring-4070')
ROOT = HOST/'learning/A-scale-20261109-v1'
CID = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def host(p): return HOST/Path(p).relative_to('/workspace')


def main():
    info=json.loads(subprocess.check_output(['docker','inspect',CID],text=True))[0]
    assert info['Id']==CID and info['Name']=='/aiworker_pouring_4070'
    assert not info['State']['Running']
    m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json')
    assert sha(ROOT/'plan.json')==m['plan_sha256']
    assert sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(host(p))==h for p,h in m['protected_parent_files'].items())
    assert all(sha(HOST/'src'/n)==h for n,h in m['runtime_sources'].items())
    cases=[]
    for s in plan['pilot']:
        matches=[p.parent for p in (HOST/'outputs/fountain').glob('*/*SC_A_*/parameters.json') if read(p)['name']==s['name']]
        assert len(matches)<=1
        row=dict(name=s['name'],status='unstarted')
        if matches:
            p=matches[0]
            complete=all((p/n).exists() for n in ('episode.hdf5','metrics.json','samples.json','trajectory.npz'))
            row.update(status='complete' if complete else 'interrupted_partial',path=str(p),
                       files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in p.iterdir() if f.is_file()})
            if complete:
                samples=read(p/'samples.json');f=samples[-1]
                row.update(final_height_fraction=f['cup_fill_fraction'],lost_particles=round(f['ever_table_ml']/.0018**3/1e6),
                           final_spilled_nominal_ml=f['spilled_ml'],first_sample={k:samples[0][k] for k in ('t','cup_ml','spilled_ml','airborne_ml','ever_table_ml')})
        cases.append(row)
    log=subprocess.check_output(['journalctl','-u','docker.service','--since','2026-11-09 16:59:00 UTC',
                                 '--until','2026-11-09 17:02:00 UTC','--no-pager'],text=True)
    lines=[l for l in log.splitlines() if CID in l or any(x in l for x in ('Stopping Docker','Stopped Docker','Starting Docker','Started Docker','Daemon shutdown complete'))]
    report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),container=CID,state=info['State'],
                cause='Host Docker daemon restart interrupted the second pilot. Initiator not established.',
                docker_journal=lines,cases=cases,parent_evidence_unchanged=True,
                checkpoint_sha256=m['checkpoint_sha256'],plan_sha256=m['plan_sha256'],
                simulator_restarted=False,other_jobs_modified=False,
                scope='Calibration evidence only. No new training or autonomous task result.')
    with (ROOT/'pilot-r0-interruption.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(cases=[{k:r[k] for k in ('name','status')} for r in cases],parent_evidence_unchanged=True)))


if __name__=='__main__':main()
