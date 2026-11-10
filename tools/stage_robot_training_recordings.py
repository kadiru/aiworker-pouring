"""Copy and checksum-verify owned recordings; never delete or switch sources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

UUID = 'e7cfe6a5-0495-4ada-9f12-8ed2033be904'
MOUNT = Path('/mnt/robot-training')
ROOT = Path('/home/kadir/aiworker-pouring-4070')
SOURCE = ROOT / 'outputs'
PROJECT = MOUNT / 'aiworker-pouring'
DEST = PROJECT / 'outputs'
AUDIT = ROOT / 'storage-migration-20261110'
CID = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def guard():
    actual = subprocess.check_output(['findmnt', '-n', '-M', str(MOUNT), '-o', 'UUID'], text=True).strip()
    assert actual == UUID, 'Required data partition is not mounted'
    container = json.loads(subprocess.check_output(['docker', 'inspect', CID], text=True))[0]
    assert not container['State']['Running'], 'Owned simulator must be stopped during migration'
    assert not SOURCE.is_symlink() and not os.path.ismount(SOURCE), 'Source already redirected'
    assert not PROJECT.is_symlink() and not DEST.is_symlink(), 'Destination must not be redirected'


def inventory(root):
    result = {}
    def error(exc):
        raise exc
    for directory, directories, files in os.walk(root, followlinks=False, onerror=error):
        for name in sorted(directories + files):
            p = Path(directory) / name
            s = p.lstat()
            kind = 'file' if stat.S_ISREG(s.st_mode) else 'dir' if stat.S_ISDIR(s.st_mode) else 'link' if stat.S_ISLNK(s.st_mode) else 'unsupported'
            assert kind != 'unsupported', str(p)
            result[str(p.relative_to(root))] = dict(kind=kind, size=s.st_size if kind == 'file' else 0,
                mtime_ns=s.st_mtime_ns, mode=stat.S_IMODE(s.st_mode),
                link=os.readlink(p) if kind == 'link' else None)
    return result


def main():
    guard()
    start = time.monotonic()
    before = inventory(SOURCE)
    total = sum(r['size'] for r in before.values())
    free = shutil.disk_usage(MOUNT).free
    expected = dict(source=str(SOURCE),destination=str(DEST),uuid=UUID,source_inventory_sha256=digest(before),
                    source_bytes=total,source_entries=len(before),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    AUDIT.mkdir(exist_ok=True)
    plan = AUDIT / 'stage-plan.json'
    if plan.exists():
        assert json.loads(plan.read_text()) == expected, 'Source or staging plan changed; inspect before resuming'
    else:
        assert not DEST.exists(), 'Destination already exists without a staging plan'
        assert free > total + 57 * 1024**3, 'Keep the planned next-session storage headroom'
        plan.write_text(json.dumps(expected, indent=2) + '\n')
        (AUDIT / 'source-inventory.json').write_text(json.dumps(before, sort_keys=True, separators=(',', ':')) + '\n')
    PROJECT.mkdir(exist_ok=True)
    flags = ['-aH', '--no-owner', '--no-group', '--checksum-choice=sha1']
    command = ['rsync', *flags, '--whole-file', '--info=stats2', str(SOURCE) + '/', str(DEST) + '/']
    print(json.dumps(dict(phase='copy',source_bytes=total,entries=len(before),available_bytes=free)), flush=True)
    with (AUDIT / 'copy.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    guard()
    assert inventory(SOURCE) == before, 'Source changed during copy'
    subprocess.run(['sync', '-f', str(DEST)], check=True)
    print(json.dumps(dict(phase='checksum_verification',algorithm='SHA1',files=sum(r['kind']=='file' for r in before.values()))), flush=True)
    verify = subprocess.run(['rsync', *flags, '--checksum', '--dry-run', '--delete', '--itemize-changes',
                             str(SOURCE) + '/', str(DEST) + '/'], text=True, capture_output=True)
    (AUDIT / 'checksum-dry-run.log').write_text(verify.stdout + verify.stderr)
    assert verify.returncode == 0 and not verify.stdout.strip() and not verify.stderr.strip(), 'Checksum or metadata differences found'
    guard()
    assert inventory(SOURCE) == before, 'Source changed during verification'
    report = dict(expected,copy_verified=True,checksum_algorithm='SHA1 via full rsync checksum dry run',
                  verification_differences=0,regular_files=sum(r['kind']=='file' for r in before.values()),
                  owner_policy='Destination owned by kadir; contents, modes, timestamps, links verified; original ownership untouched.',
                  seconds=time.monotonic()-start,root_disk_free_bytes=shutil.disk_usage(ROOT).free,
                  destination_free_bytes=shutil.disk_usage(MOUNT).free,source_deleted=False,path_switch_applied=False,
                  scope='Verified copy only. Original paths still use the original disk. No jobs started or stopped; no originals removed.')
    (AUDIT / 'stage-completion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
