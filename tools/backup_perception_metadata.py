"""SHA-verified scalar backup; native HDF5/trajectories remain on Ubuntu."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
NATIVE='/home/kadir/aiworker-pouring-4070/learning/A-perception-20261108-v1'
CODE=r'''
import hashlib,json,sys,tarfile
from pathlib import Path
phase=sys.argv[1];assert phase.replace('-','').isalnum()
root=Path('/workspace/learning/A-perception-20261108-v1');outputs=Path('/workspace/outputs/fountain')
report=json.loads((root/'recording-audit.json').read_text());assert not report['started_incomplete']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
entries=[];archive=root/('scalar-backup-'+phase+'.tar.gz')
with tarfile.open(archive,'x:gz') as tar:
 for record in report['records']:
  folder=Path(record['path']).parent;relative=folder.relative_to(outputs)
  assert len(relative.parts)==2 and 'PV_A_' in relative.name
  for name,digest in record['sidecars'].items():
   assert Path(name).name==name
   p=folder/name;assert sha(p)==digest
   tar.add(p,arcname=str(relative/name),recursive=False)
   entries.append(dict(relative_path=str(relative/name),sha256=digest,bytes=p.stat().st_size))
manifest=dict(phase=phase,scope='Audited scalar sidecars only. Original native HDF5, particle trajectories and images are not fully backed up locally.',
 entries=entries,recordings=len(report['records']),archive_sha256=sha(archive),audit_sha256=sha(root/'recording-audit.json'))
(root/('scalar-backup-'+phase+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(recordings=len(report['records']),sidecars=len(entries),archive_sha256=manifest['archive_sha256'])))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True);a=parser.parse_args();assert a.phase.replace('-','').isalnum()
    data=PROJECT/'data/material-A-perception';data.mkdir(parents=True,exist_ok=True)
    print(subprocess.check_output(['ssh','-o','BatchMode=yes','kfuUbuntu','docker','exec','-i',CID,'/isaac-sim/python.sh','-',a.phase],input=CODE,text=True))
    stem='scalar-backup-'+a.phase
    for suffix in ('.tar.gz','.json'):
        subprocess.run(['scp','-q','kfuUbuntu:'+NATIVE+'/'+stem+suffix,str(data/(stem+suffix))],check=True)
    manifest=json.loads((data/(stem+'.json')).read_text());assert sha(data/(stem+'.tar.gz'))==manifest['archive_sha256']
    expected={r['relative_path']:r for r in manifest['entries']}
    with tarfile.open(data/(stem+'.tar.gz')) as tar:
        assert {m.name for m in tar.getmembers()}==set(expected)
        for member in tar.getmembers():
            name=Path(member.name);assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            payload=tar.extractfile(member).read();identity=expected[member.name]
            assert len(payload)==identity['bytes'] and hashlib.sha256(payload).hexdigest()==identity['sha256']
            dest=data/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==identity['sha256']
            else:dest.write_bytes(payload)
    report=dict(**manifest,verified_local_sidecars=len(expected),archive_bytes=(data/(stem+'.tar.gz')).stat().st_size)
    (PROJECT/'evidence/perception-A'/(stem+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(verified_local_sidecars=len(expected),recordings=manifest['recordings'])))
if __name__=='__main__':main()
