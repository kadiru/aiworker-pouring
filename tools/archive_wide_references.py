"""Losslessly compress verified local reference/reviewed development copies."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT=Path(__file__).resolve().parents[1]
DATA=PROJECT/'data/material-A-wide'
ZSTD='/opt/homebrew/bin/zstd'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def archive_identity(path):
    """Return the original byte count/hash by streaming a checked decompression."""
    proc=subprocess.Popen([ZSTD,'-q','-d','-c','--',str(path)],stdout=subprocess.PIPE)
    h=hashlib.sha256();size=0
    try:
        for chunk in iter(lambda:proc.stdout.read(1024*1024),b''):
            h.update(chunk);size+=len(chunk)
        proc.stdout.close();assert proc.wait()==0,'Lossless archive decode failed'
    except BaseException:
        proc.kill();proc.wait();raise
    return size,h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    p.add_argument('--development-rounds',type=int,nargs='+')
    a=p.parse_args();assert a.output.resolve().is_relative_to(PROJECT) and not a.output.exists()
    manifest=json.loads(a.manifest.read_text());records=[r for r in manifest['files'] if r['name'].startswith('T30_A_demo_')]
    if a.development_rounds is None:
        assert len(records)==37
    else:
        assert len(set(a.development_rounds))==len(a.development_rounds)
        reviewed={}
        for number in a.development_rounds:
            review=json.loads((PROJECT/f'evidence/wide-A/development-r{number}-visual/manifest.json').read_text())
            assert review['manual_review']=='inspected' and len(review['cases'])==8
            for c in review['cases']:
                assert c['manual_review']=='inspected' and c['name'].startswith('XW30_A_dev_') and c['name'].endswith(f'_r{number}')
                reviewed[c['name']]=c['source_sha256']
        records=[r for r in manifest['files'] if r['name'] in reviewed]
        assert len(records)==8*len(a.development_rounds)
        assert all(r['sha256']==reviewed[r['name']] for r in records)
    if a.limit is not None:assert a.limit>0;records=records[:a.limit]
    rows=[]
    for r in records:
        relative=Path(r['relative_path']);assert not relative.is_absolute() and '..' not in relative.parts
        source=DATA/relative;archive=source.with_suffix(source.suffix+'.zst');created=False
        if source.exists():
            assert source.stat().st_size==r['bytes'] and digest(source)==r['sha256']
            assert not archive.exists(),'Inspect duplicate raw/archive copies before proceeding'
            partial=archive.with_suffix('.partial.zst');assert not partial.exists()
            subprocess.run([ZSTD,'-q','-T1','-6','--no-progress','--check','-o',str(partial),'--',str(source)],check=True)
            assert archive_identity(partial)==(r['bytes'],r['sha256'])
            partial.replace(archive)
            # Only this newly generated local duplicate is replaced. The
            # verified byte-identical archive and native original both remain.
            source.unlink();created=True
        else:
            assert archive_identity(archive)==(r['bytes'],r['sha256'])
        rows.append(dict(name=r['name'],relative_path=r['relative_path'],original_bytes=r['bytes'],
            original_sha256=r['sha256'],archive=str(archive.relative_to(DATA)),archive_bytes=archive.stat().st_size,
            archive_sha256=digest(archive),created_this_pass=created))
        print(json.dumps({k:rows[-1][k] for k in ('name','original_bytes','archive_bytes','created_this_pass')}),flush=True)
    report=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_index_sha256=digest(a.manifest),
        records=rows,bytes_saved=sum(r['original_bytes']-r['archive_bytes'] for r in rows),
        development_rounds=a.development_rounds,
        scope='Lossless local copies of closed protocol-17 references or explicitly reviewed development recordings. Every decompressed byte count and SHA matches the native index. Native originals, test recordings and model/runtime state are unchanged.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),bytes_saved=report['bytes_saved'])),flush=True)


if __name__=='__main__':main()
