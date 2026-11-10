"""Switch verified recordings to their new disk while keeping existing paths.

Default: read-only preflight. --apply requires sudo and installs one bind mount.
--remove-verified-originals additionally removes only the verified old copy,
after an independent checksum comparison and a CPU-only Docker read check.
No formatting, model change, simulator start, or other-job control is performed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import configure_robot_training_mount as mount
import stage_robot_training_recordings as s

BACKUP = s.ROOT / 'outputs.pre-robot-training-20261110'
ENTRY = f'{s.DEST} {s.SOURCE} none bind,nofail,x-systemd.requires-mounts-for={s.MOUNT} 0 0'
PROBE = 'fountain/water-preset.json'


def proposed_fstab(text):
    if mount.proposed_fstab(text) != text:
        raise RuntimeError('Install the verified primary data-partition mount first.')
    matches = []
    for line in text.splitlines():
        fields = line.split('#', 1)[0].split()
        if fields and (fields[0] == str(s.DEST) or (len(fields) > 1 and fields[1] == str(s.SOURCE))):
            matches.append(fields)
    if matches:
        if matches == [ENTRY.split()]:
            return text
        raise RuntimeError('An existing recordings mount conflicts with this proposal.')
    return text.rstrip('\n') + '\n\n# AIWorker recordings; preserve existing host and container paths\n' + ENTRY + '\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def container_stopped():
    c = json.loads(subprocess.check_output(['docker', 'inspect', s.CID], text=True))[0]
    assert not c['State']['Running'], 'Owned simulator is running; no automatic stop is permitted'
    assert c['Config']['Labels'].get('owner') == 'aiworker-pouring'
    assert any(m['Source'] == str(s.ROOT) and m['Destination'] == '/workspace' for m in c['Mounts'])
    return c


def no_nested_mounts(path):
    prefix = str(path)
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        target = line.split()[4]
        assert target != prefix and not target.startswith(prefix + '/'), f'Unexpected mount under {path}'


def preflight():
    s.guard()
    assert s.ROOT.resolve() == s.ROOT and s.SOURCE.resolve() == s.SOURCE and s.DEST.resolve() == s.DEST
    assert not BACKUP.exists(), 'Original-copy backup path already exists; inspect saved receipt'
    no_nested_mounts(s.SOURCE)
    report = json.loads((s.AUDIT / 'stage-completion.json').read_text())
    before = json.loads((s.AUDIT / 'source-inventory.json').read_text())
    assert report['copy_verified'] and report['verification_differences'] == 0
    assert report['source'] == str(s.SOURCE) and report['destination'] == str(s.DEST) and report['uuid'] == s.UUID
    assert report['source_inventory_sha256'] == s.digest(before)
    assert s.inventory(s.SOURCE) == before and s.inventory(s.DEST) == before
    assert (s.SOURCE / PROBE).is_file() and sha(s.SOURCE / PROBE) == sha(s.DEST / PROBE)
    proposed_fstab(mount.FSTAB.read_text())
    container_stopped()
    return before, report


def checksum_equal(source, target, log):
    result = subprocess.run(['rsync', '-aHcn', '--no-owner', '--no-group', '--checksum-choice=sha1',
        '--delete', '--itemize-changes', str(source) + '/', str(target) + '/'], text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and not result.stdout.strip() and not result.stderr.strip(), 'Content or metadata comparison failed'


def verify_alias_and_container(image, expected_probe):
    assert subprocess.check_output(['findmnt', '-n', '-M', str(s.SOURCE), '-o', 'UUID'], text=True).strip() == s.UUID
    assert os.path.samefile(s.SOURCE, s.DEST), 'Old path does not address the exact new directory'
    command = ['docker', 'run', '--rm', '--pull=never', '--runtime=runc', '--network=none', '--read-only',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user=1000:1000', '--cpus=0.25', '--memory=128m',
        '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--entrypoint=/usr/bin/sha256sum',
        '--mount', f'type=bind,source={s.ROOT},target=/workspace,readonly', image, '/workspace/outputs/' + PROBE]
    output = subprocess.check_output(command, text=True).strip()
    assert output.split()[0] == expected_probe, 'Read-only container view differs from verified source'
    return output


def apply(remove_originals):
    if os.geteuid() != 0:
        raise RuntimeError('Run --apply with sudo in your Ubuntu terminal.')
    before, staged = preflight()
    print('Checking every source and destination file before switching paths...', flush=True)
    checksum_equal(s.SOURCE, s.DEST, s.AUDIT / 'cutover-before-checksum.log')
    assert s.inventory(s.SOURCE) == before and s.inventory(s.DEST) == before
    c = container_stopped()
    source_stat = s.SOURCE.stat()
    probe_hash = sha(s.SOURCE / PROBE)
    original = mount.FSTAB.read_text()
    updated = proposed_fstab(original)
    metadata = mount.FSTAB.stat()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    fstab_backup = Path(f'/etc/fstab.aiworker-recordings-{stamp}.bak')
    with fstab_backup.open('x') as f:
        f.write(original)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(fstab_backup, 0o644)
    subprocess.run(['sync', '-f', str(s.DEST)], check=True)
    renamed = False
    mounted = False
    try:
        mount.atomic_fstab(updated, metadata, original)
        s.SOURCE.rename(BACKUP)
        renamed = True
        s.SOURCE.mkdir(mode=0o755)
        os.chown(s.SOURCE, source_stat.st_uid, source_stat.st_gid)
        subprocess.run(['systemctl', 'daemon-reload'], check=True)
        subprocess.run(['mount', '--', str(s.SOURCE)], check=True)
        mounted = True
        docker_probe = verify_alias_and_container(c['Image'], probe_hash)
        assert s.inventory(s.SOURCE) == before and s.inventory(BACKUP) == before
        container_stopped()
    except Exception:
        if mounted:
            subprocess.run(['umount', '--', str(s.SOURCE)], check=True)
        if renamed:
            if s.SOURCE.exists():
                s.SOURCE.rmdir()  # Empty mount point only; never recursively remove it.
            BACKUP.rename(s.SOURCE)
        if mount.FSTAB.read_text() == updated:
            mount.atomic_fstab(original, metadata, updated)
            subprocess.run(['systemctl', 'daemon-reload'], check=False)
        raise
    receipt = dict(mounted=True,uuid=s.UUID,source_alias=str(s.SOURCE),data_directory=str(s.DEST),
        source_inventory_sha256=s.digest(before),regular_files=staged['regular_files'],source_bytes=staged['source_bytes'],
        fstab_backup=str(fstab_backup),retained_original_copy=str(BACKUP),
        original_inode=source_stat.st_ino,original_device=source_stat.st_dev,
        original_copy_removed=False,docker_probe_sha256=probe_hash,docker_probe_output=docker_probe,
        helper_sha256=sha(Path(__file__)),scope='Verified path switch; original simulator remains stopped; no model/source change or other-job control.')
    output = s.AUDIT / 'cutover-completion.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    if remove_originals:
        # This region never rolls back to a partially removed original tree.
        # The complete, checksum-verified new copy is already the active path.
        print('Verifying the mounted path before releasing the old duplicate...', flush=True)
        checksum_equal(BACKUP, s.SOURCE, s.AUDIT / 'cutover-after-checksum.log')
        assert s.inventory(BACKUP) == before and s.inventory(s.SOURCE) == before
        assert not BACKUP.is_symlink()
        assert (BACKUP.stat().st_dev, BACKUP.stat().st_ino) == (source_stat.st_dev, source_stat.st_ino)
        no_nested_mounts(BACKUP)
        assert os.path.samefile(s.SOURCE, s.DEST)
        container_stopped()
        subprocess.run(['sync', '-f', str(s.DEST)], check=True)
        shutil.rmtree(BACKUP)
        receipt['original_copy_removed'] = True
        receipt['retained_original_copy'] = None
    receipt['root_free_bytes'] = shutil.disk_usage(s.ROOT).free
    receipt['data_free_bytes'] = shutil.disk_usage(s.MOUNT).free
    receipt['required_preflight'] = 'python3 /home/kadir/aiworker-pouring-4070/tools/verify_robot_training_storage.py --minimum-free-gib 57'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--remove-verified-originals', action='store_true')
    args = parser.parse_args()
    if args.remove_verified_originals and not args.apply:
        parser.error('--remove-verified-originals also requires --apply')
    if args.apply:
        apply(args.remove_verified_originals)
    else:
        _, report = preflight()
        print(json.dumps(dict(ready=True,administrator=os.geteuid()==0,source_bytes=report['source_bytes'],
                              fstab_entry=ENTRY,scope='Read-only metadata preflight; apply repeats complete checksums.')))


if __name__ == '__main__':
    main()
