"""Verify one temporary native recording at a time; preserve remote originals."""
from contextlib import contextmanager
from pathlib import Path,PurePosixPath
import hashlib,subprocess,tempfile

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

@contextmanager
def verified_recording(identity):
    rel=PurePosixPath(identity['relative_path'])
    assert not rel.is_absolute() and '..' not in rel.parts and rel.name=='episode.hdf5'
    with tempfile.TemporaryDirectory(prefix='aiworker-finish-media-') as folder:
        path=Path(folder)/'episode.hdf5'
        subprocess.run(['scp','-q','kfuUbuntu:/home/kadir/aiworker-pouring-4070/outputs/fountain/'+str(rel),str(path)],check=True)
        assert path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256']
        yield path
