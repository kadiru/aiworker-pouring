"""Index only newly closed tests for incremental backup; never open active files."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


BASE = Path('/workspace/outputs/fountain')
STUDY = Path('/workspace/learning/A-workspace-20261104-v1')


def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return dict(relative_path=str(p.relative_to(BASE)), bytes=p.stat().st_size, sha256=h.hexdigest())


def main(milestone):
    start={5:0,15:5,25:15}[milestone]
    frozen=json.loads((STUDY/'final-selection.json').read_text());sha=frozen['selected_checkpoint_sha256']
    snap=Path('/workspace/study-snapshots')/('W30-'+sha[:16])/f'runtime-after-test{milestone:02}.json'
    witness=json.loads(snap.read_text());assert witness['selected_checkpoint_sha256']==sha and witness['sources']==frozen['runtime_sources']
    names=frozen['expected_test_names'][start:milestone]
    assert set(names)<=set(witness['completed_test_cases_at_observation'])
    files=[];sidecars=[]
    for name in names:
        folders=list(BASE.glob('*/*-'+name+'-*'));assert len(folders)==1,name
        folder=folders[0];metrics=json.loads((folder/'metrics.json').read_text())
        assert json.loads((folder/'parameters.json').read_text())['name']==name and metrics['policy']['checkpoint_sha256']==sha
        assert not (folder/'episode.partial.hdf5').exists()
        files.append(dict(name=name,**entry(folder/'episode.hdf5')))
        sidecars.extend(entry(folder/n) for n in ('parameters.json','metrics.json','samples.json','policy-decisions.json'))
    result=dict(indexed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_sha256=sha,
        scope='Byte-level backup of this closed test block only; full recording validation follows the complete evaluation.',
        files=files,sidecar_files=sidecars,partial_files=[],partial_sidecar_files=[])
    path=STUDY/f'backup-test-block-{start:02}-{milestone-1:02}.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['files']==files and old['sidecar_files']==sidecars
    else:
        with path.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('milestone',type=int,choices=(5,15,25));main(p.parse_args().milestone)
