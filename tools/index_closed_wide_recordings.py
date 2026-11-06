"""Host-side byte index of closed protocol-17 recordings; no active HDF5 reads."""
import argparse,datetime,hashlib,json
from pathlib import Path

BASE=Path('/home/kadir/aiworker-pouring-4070/outputs/fountain')


def entry(p):
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return dict(relative_path=str(p.relative_to(BASE)),bytes=after.st_size,sha256=h.hexdigest())


def main(output):
    assert not output.exists();files=[];sidecars=[];names=set();pending=[]
    for prefix in ('T30_A_demo_','XW30_A_'):
        for p in sorted(BASE.glob('*/*'+prefix+'*/parameters.json')):
            s=json.loads(p.read_text())
            if s.get('protocol')!=17:continue
            name=s['name'];assert name not in names;names.add(name);folder=p.parent;ep=folder/'episode.hdf5'
            if not ep.exists() or not (folder/'metrics.json').exists() or (folder/'episode.partial.hdf5').exists():
                pending.append(name);continue
            files.append(dict(name=name,**entry(ep)))
            required=('parameters.json','metrics.json','samples.json')+(('policy-decisions.json',) if s['policy_evaluation'] else ())
            sidecars.extend(entry(folder/n) for n in required)
    report=dict(indexed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,sidecar_files=sidecars,
        partial_files=[],partial_sidecar_files=[],pending_attempts=pending,
        scope='Closed-file byte hashes only, not recording validation or task success. Full native integrity audit is separate.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(closed=len(files),sidecars=len(sidecars),pending=pending)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
