"""Copy and verify audited scalar sidecars; original HDF5 recordings stay native."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
NATIVE='/home/kadir/aiworker-pouring-4070/learning/A-size-20261108-v1'
CODE=r'''
import hashlib,json,tarfile
from pathlib import Path
r=Path('/workspace/learning/A-size-20261108-v1');outputs=Path('/workspace/outputs/fountain')
report=json.loads((r/'stage-results.json').read_text())
assert report['development_closed'] and report['all_seed_calibration_contained']
assert len(report['integrity'])==24
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
entries=[]
with tarfile.open(r/'scalar-backup.tar.gz','w:gz') as tar:
 for record in report['integrity']:
  folder=Path(record['path']).parent;relative=folder.relative_to(outputs)
  assert len(relative.parts)==2 and 'SZ_A_' in relative.name
  for name,digest in record['sidecars'].items():
   assert Path(name).name==name
   p=folder/name;assert sha(p)==digest
   tar.add(p,arcname=str(relative/name),recursive=False)
   entries.append(dict(relative_path=str(relative/name),sha256=digest,bytes=p.stat().st_size))
manifest=dict(scope='All audited scalar sidecars, including failed grid and recovery attempts. Original HDF5 files remain on Ubuntu; no full raw local backup is claimed.',
 entries=entries,recordings=24,archive_sha256=sha(r/'scalar-backup.tar.gz'),report_sha256=sha(r/'stage-results.json'))
(r/'scalar-backup-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(recordings=24,sidecars=len(entries),archive_sha256=manifest['archive_sha256'])))
'''


def main():
    data=PROJECT/'data/material-A-size';data.mkdir(exist_ok=True)
    command=['ssh','-o','BatchMode=yes','kfuUbuntu','docker','exec','-i',CID,'/isaac-sim/python.sh','-']
    print(subprocess.check_output(command,input=CODE,text=True))
    subprocess.run(['scp','-q','kfuUbuntu:'+NATIVE+'/scalar-backup.tar.gz',str(data/'scalar-backup.tar.gz')],check=True)
    subprocess.run(['scp','-q','kfuUbuntu:'+NATIVE+'/scalar-backup-manifest.json',str(data/'scalar-backup-manifest.json')],check=True)
    manifest=json.loads((data/'scalar-backup-manifest.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(data/'scalar-backup.tar.gz')==manifest['archive_sha256']
    expected={r['relative_path']:r for r in manifest['entries']}
    with tarfile.open(data/'scalar-backup.tar.gz') as tar:
        members=tar.getmembers();assert {m.name for m in members}==set(expected)
        for member in members:
            name=Path(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            target=data/name;target.parent.mkdir(parents=True,exist_ok=True)
            payload=tar.extractfile(member).read();identity=expected[member.name]
            assert len(payload)==identity['bytes'] and hashlib.sha256(payload).hexdigest()==identity['sha256']
            if target.exists():assert sha(target)==identity['sha256']
            else:target.write_bytes(payload)
    report=dict(**manifest,verified_local_sidecars=len(expected),archive_bytes=(data/'scalar-backup.tar.gz').stat().st_size)
    (PROJECT/'evidence/size-A/scalar-backup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(verified_local_sidecars=len(expected),recordings=24)))


if __name__=='__main__':main()
