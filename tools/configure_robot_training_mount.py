"""Check, or explicitly apply, the user's existing ext4 storage mount.

No partitioning, formatting, data migration, job control, or data deletion.
Run without arguments to inspect; --apply requires Ubuntu administrator rights.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

UUID = 'e7cfe6a5-0495-4ada-9f12-8ed2033be904'
TARGET = Path('/mnt/robot-training')
OLD_TARGET = Path('/media/kadir/robot-training')
FSTAB = Path('/etc/fstab')
ENTRY = f'UUID={UUID} {TARGET} ext4 defaults,noatime,nofail,x-systemd.device-timeout=10s 0 2'


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def proposed_fstab(text):
    matching = []
    for line in text.splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if fields[0] in (f'UUID={UUID}', f'/dev/disk/by-uuid/{UUID}', '/dev/nvme0n1p8') or (
                len(fields) > 1 and fields[1] == str(TARGET)):
            matching.append(fields)
    if matching:
        if matching == [ENTRY.split()]:
            return text
        raise RuntimeError('Existing storage entry differs; inspect it before proceeding.')
    return text.rstrip('\n') + '\n\n# Dedicated robot-training data partition\n' + ENTRY + '\n'


def inspect():
    link = Path('/dev/disk/by-uuid') / UUID
    if not link.exists():
        raise RuntimeError('The verified partition UUID is not present.')
    device = link.resolve()
    data = json.loads(run('lsblk', '-b', '-J', '-o', 'PATH,FSTYPE,LABEL,UUID,SIZE,MOUNTPOINTS', str(device)))
    block = data['blockdevices'][0]
    if block['uuid'] != UUID or block['fstype'] != 'ext4' or block['label'] != 'robot-training':
        raise RuntimeError('Partition identity or filesystem differs from the verified disk.')
    mounts = [p for p in block['mountpoints'] if p]
    if len(mounts) > 1 or any(Path(p) not in (TARGET, OLD_TARGET) for p in mounts):
        raise RuntimeError(f'Unexpected mounts: {mounts}')
    if TARGET.is_symlink():
        raise RuntimeError('Requested mount point must not be a symlink.')
    if TARGET.exists() and str(TARGET) not in mounts and any(TARGET.iterdir()):
        raise RuntimeError('Requested mount point contains files; refusing to cover them.')
    text = FSTAB.read_text()
    return dict(device=str(device),uuid=UUID,filesystem='ext4',partition_bytes=block['size'],
                current_mounts=mounts,target=str(TARGET),fstab_entry=ENTRY,
                fstab_change_needed=proposed_fstab(text) != text,
                administrator=os.geteuid() == 0,scope=__doc__.strip())


def atomic_fstab(text, metadata, expected):
    fd, name = tempfile.mkstemp(prefix='.fstab.robot-training-', dir='/etc')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
            os.fchmod(f.fileno(), stat.S_IMODE(metadata.st_mode))
            os.fchown(f.fileno(), metadata.st_uid, metadata.st_gid)
        subprocess.run(['findmnt', '--verify', '--tab-file', name], check=True)
        if FSTAB.read_text() != expected:
            raise RuntimeError('fstab changed concurrently; refusing to overwrite it.')
        os.replace(name, FSTAB)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def apply():
    if os.geteuid() != 0:
        raise RuntimeError('--apply requires sudo in your Ubuntu terminal.')
    before = inspect()
    original = FSTAB.read_text()
    updated = proposed_fstab(original)
    metadata = FSTAB.stat()
    current = before['current_mounts'][0] if before['current_mounts'] else None
    if not TARGET.exists():
        TARGET.mkdir(mode=0o755)
    backup = None
    if updated != original:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = Path(f'/etc/fstab.robot-training-{stamp}.bak')
        with backup.open('x') as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(backup, stat.S_IMODE(metadata.st_mode))
        atomic_fstab(updated, metadata, original)
    try:
        subprocess.run(['systemctl', 'daemon-reload'], check=True)
        if current and current != str(TARGET):
            subprocess.run(['umount', '--', current], check=True)
        if current != str(TARGET):
            subprocess.run(['mount', '--', str(TARGET)], check=True)
        actual = run('findmnt', '-n', '-M', str(TARGET), '-o', 'UUID')
        if actual != UUID:
            raise RuntimeError('Mounted UUID differs; refusing any further changes.')
    except Exception:
        # Restore only this helper's exact fstab version; never overwrite a concurrent edit.
        if updated != original and FSTAB.read_text() == updated:
            atomic_fstab(original, metadata, updated)
            subprocess.run(['systemctl', 'daemon-reload'], check=False)
        raise
    print(json.dumps(dict(mounted=True,uuid=actual,target=str(TARGET),fstab_backup=str(backup) if backup else None,
                          available_bytes=shutil.disk_usage(TARGET).free,
                          note='No recordings moved or deleted. Read/write verification as kadir is the next check.')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        apply()
    else:
        print(json.dumps(inspect(), indent=2))


if __name__ == '__main__':
    main()
