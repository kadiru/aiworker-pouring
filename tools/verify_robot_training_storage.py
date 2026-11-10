"""Required storage preflight before starting migrated AIWorker collection jobs."""
import argparse
import json
import os
import shutil
import subprocess
from stage_robot_training_recordings import UUID, MOUNT, SOURCE, DEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-free-gib', type=float, default=0)
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error('Minimum free space cannot be negative')
    for path in (MOUNT, SOURCE):
        actual = subprocess.check_output(['findmnt', '-n', '-M', str(path), '-o', 'UUID'], text=True).strip()
        if actual != UUID:
            raise RuntimeError(f'Required data partition is not mounted at {path}; do not start collection')
    if not os.path.samefile(SOURCE, DEST):
        raise RuntimeError('Recordings path does not address the migrated directory')
    free = shutil.disk_usage(MOUNT).free
    if free < args.minimum_free_gib * 1024**3:
        raise RuntimeError('Insufficient free space for the requested collection budget')
    print(json.dumps(dict(ready=True,uuid=UUID,recordings=str(DEST),compatible_path=str(SOURCE),free_gib=free/1024**3)))


if __name__ == '__main__':
    main()
