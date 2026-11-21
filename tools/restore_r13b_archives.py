"""Verify archived recordings on the Mac or restore them to their recorded Ubuntu paths."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from compress_sized_recordings import archive_identity, sha

PROJECT = Path(__file__).resolve().parents[1]
STUDY = "wide-mixed-r13b"
ARCHIVES = Path("/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1")/STUDY/"recording-archives"
RECEIPT = PROJECT/"evidence/bottle-size-A"/(STUDY+"-mac-offload.json")


def inventory():
    records = []
    for receipt in sorted(RECEIPT.parent.glob(STUDY+"-*-recording-archives.json")):
        batch = json.loads(receipt.read_text())
        assert batch["complete"] and batch["report_sha256"]==sha(receipt.with_name(receipt.name.replace("-recording-archives.json","-results.json")))
        for record in batch["records"]:
            path=Path(record["archive"])
            if path.parent==ARCHIVES:
                assert path.name.startswith("SA_A_wide_mixed_r13b_") and path.name.endswith(".hdf5.zst")
                records.append(dict(record,local=str(Path("data")/STUDY/path.name)))
    assert len(records)==len({r["archive"] for r in records})==24
    return sorted(records,key=lambda r:r["archive"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--restore",metavar="ARCHIVE_BASENAME")
    parser.add_argument("--write-receipt",action="store_true")
    args=parser.parse_args()
    records=inventory()
    if args.restore:
        records=[r for r in records if Path(r["archive"]).name==args.restore]
        assert len(records)==1 and not args.write_receipt
    for r in records:
        local=PROJECT/r["local"]
        assert local.stat().st_size==r["archive_bytes"] and sha(local)==r["archive_sha256"]
        assert archive_identity(local)==(r["original_bytes"],r["original_sha256"])
    if args.write_receipt:
        RECEIPT.write_text(json.dumps(dict(records=records,verified_native_bytes=sum(r["original_bytes"] for r in records),
                               archive_bytes=sum(r["archive_bytes"] for r in records)),indent=2)+"\n")
    if args.restore:
        r=records[0]
        # Upload privately alongside the destination and install only verified bytes.
        program = """from pathlib import Path
import hashlib, os, shutil, sys, tempfile
p=Path(sys.argv[1]); n=int(sys.argv[2]); digest=sys.argv[3]
assert p.parent.resolve()==p.parent and p.parent.is_dir()
with tempfile.NamedTemporaryFile(dir=p.parent) as f:
    assert shutil.disk_usage(p.parent).free>n
    shutil.copyfileobj(sys.stdin.buffer,f);f.flush()
    h=hashlib.sha256()
    with open(f.name,'rb') as check:
        for b in iter(lambda:check.read(4194304),b''):h.update(b)
    assert f.tell()==n and h.hexdigest()==digest
    if not p.exists():os.link(f.name,p)
    with p.open('rb') as check:
        h=hashlib.sha256()
        for b in iter(lambda:check.read(4194304),b''):h.update(b)
    assert h.hexdigest()==digest
"""
        with (PROJECT/r["local"]).open("rb") as f:
            subprocess.run(["ssh","-o","BatchMode=yes","kfuUbuntu",shlex.join(["python3","-c",program,r["archive"],str(r["archive_bytes"]),r["archive_sha256"]])],stdin=f,check=True)
    print("Verified",len(records),"archives")


if __name__=="__main__":
    main()
