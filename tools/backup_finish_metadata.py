"""Back up audited scalar metadata; raw recordings stay native for temporary review."""
import argparse,json,subprocess
from pathlib import Path,PurePosixPath
from finish_media_source import sha

PROJECT=Path(__file__).resolve().parents[1]
DATA=PROJECT/'data/material-A-finish'

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    audit=json.loads(a.manifest.read_text());assert audit['native_checks_passed'] and not audit['pending_attempts']
    scalar_files=audit['sidecar_files']+audit.get('partial_sidecar_files',[])
    for identity in scalar_files:
        rel=PurePosixPath(identity['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix=='.json'
        target=DATA/str(rel);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary=target.with_suffix('.partial.json')
            subprocess.run(['scp','-q','kfuUbuntu:/home/kadir/aiworker-pouring-4070/outputs/fountain/'+str(rel),str(temporary)],check=True)
            assert temporary.stat().st_size==identity['bytes'] and sha(temporary)==identity['sha256']
            temporary.replace(target)
        assert target.stat().st_size==identity['bytes'] and sha(target)==identity['sha256']
    print(json.dumps(dict(verified_sidecars=len(scalar_files),native_recordings=len(audit['files']),
        scope='Scalar metadata backed up locally. Raw recordings remain on Ubuntu and are temporarily fetched and SHA-verified for media review.')))

if __name__=='__main__':main()
