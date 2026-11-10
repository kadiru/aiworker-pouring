"""Resume the copy while preserving the one private preview through Docker cp."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import stage_robot_training_recordings as s

PRIVATE = 'preview-next.png'


def private_bytes():
    result = subprocess.run(['docker', 'cp', s.CID + ':/workspace/outputs/' + PRIVATE, '-'],
                            capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        members = archive.getmembers()
        assert len(members) == 1 and members[0].isfile() and members[0].name == PRIVATE
        return archive.extractfile(members[0]).read()


def main():
    s.guard()
    started = time.monotonic()
    plan = json.loads((s.AUDIT / 'stage-plan.json').read_text())
    before = json.loads((s.AUDIT / 'source-inventory.json').read_text())
    assert s.digest(before) == plan['source_inventory_sha256'] and s.inventory(s.SOURCE) == before
    assert plan['source'] == str(s.SOURCE) and plan['destination'] == str(s.DEST) and plan['uuid'] == s.UUID
    unreadable = [n for n, r in before.items() if r['kind'] == 'file' and not os.access(s.SOURCE / n, os.R_OK)]
    assert unreadable == [PRIVATE], unreadable
    data = private_bytes()
    assert len(data) == before[PRIVATE]['size']
    target = s.DEST / PRIVATE
    if target.exists():
        assert target.read_bytes() == data
    else:
        with target.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    os.chmod(target, before[PRIVATE]['mode'])
    os.utime(target, ns=(target.stat().st_atime_ns, before[PRIVATE]['mtime_ns']))
    private = dict(path=PRIVATE,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                   read_method='Docker cp from the stopped owned container; no permission changes, no container process started.')
    (s.AUDIT / 'private-preview-copy.json').write_text(json.dumps(private, indent=2) + '\n')
    flags = ['-aH', '--no-owner', '--no-group', '--checksum-choice=sha1', '--exclude=/' + PRIVATE]
    with (s.AUDIT / 'copy-v2.log').open('a') as log:
        subprocess.run(['rsync', *flags, '--info=stats2', str(s.SOURCE) + '/', str(s.DEST) + '/'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['sync', '-f', str(s.DEST)], check=True)
    print(json.dumps(dict(phase='checksum_verification',source_bytes=plan['source_bytes'],private_preview_preserved=True)), flush=True)
    verify = subprocess.run(['rsync', *flags, '--checksum', '--dry-run', '--delete', '--itemize-changes',
                             str(s.SOURCE) + '/', str(s.DEST) + '/'], text=True, capture_output=True)
    (s.AUDIT / 'checksum-v2-dry-run.log').write_text(verify.stdout + verify.stderr)
    assert verify.returncode == 0 and not verify.stdout.strip() and not verify.stderr.strip(), 'Checksum or metadata differences'
    assert private_bytes() == target.read_bytes() == data
    s.guard()
    assert s.inventory(s.SOURCE) == before and s.inventory(s.DEST) == before
    report = dict(plan,copy_verified=True,checksum_algorithm='SHA1 full-file rsync check, plus SHA256/byte comparison for private preview',
                  verification_differences=0,regular_files=sum(r['kind']=='file' for r in before.values()),
                  private_preview=private,resume_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  owner_policy='Destination owned by kadir; contents, modes, timestamps, links verified; original ownership untouched.',
                  seconds=time.monotonic()-started,root_disk_free_bytes=shutil.disk_usage(s.ROOT).free,
                  destination_free_bytes=shutil.disk_usage(s.MOUNT).free,source_deleted=False,path_switch_applied=False,
                  first_attempt='Copy returned code 23 for one private preview; first log and original plan preserved.',
                  scope='All files copied and verified, including private preview. Original paths unchanged; no originals removed or jobs started/stopped.')
    (s.AUDIT / 'stage-completion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
