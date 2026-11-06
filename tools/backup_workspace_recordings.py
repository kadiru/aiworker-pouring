"""Copy closed study artifacts and verify every native size/SHA256 locally."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
REMOTE = '/home/kadir/aiworker-pouring-4070/outputs/fountain'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(manifest, verification, study='workspace'):
    report = json.loads(manifest.read_text())
    prefix, directory = {'workspace': ('W30_A_', 'data/material-A-workspace'),
                         'layout': ('V12_A_', 'data/material-A-layout-variety'),
                         'layout-extra': ('VX2_A_', 'data/material-A-layout-variety'),
                         'wide': (('XW30_A_', 'T30_A_demo_'), 'data/material-A-wide')}[study]
    destination = PROJECT / directory
    keys = ('files', 'sidecar_files', 'partial_files', 'partial_sidecar_files')
    items = [item for key in keys for item in report.get(key, [])]
    assert items, 'Empty integrity manifest'
    names = [item['relative_path'] for item in items]
    assert len(names) == len(set(names)), 'Duplicate artifact'
    copied = 0
    for item in items:
        relative = Path(item['relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert re.fullmatch(r'[A-Za-z0-9_./-]+', str(relative))
        assert any(p in relative.parts[1] for p in (prefix if isinstance(prefix,tuple) else (prefix,))), relative
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.download')
            subprocess.run(['scp', '-q', f'kfuUbuntu:{REMOTE}/{relative}', str(temporary)], check=True)
            assert temporary.stat().st_size == item['bytes'], str(relative)
            assert digest(temporary) == item['sha256'], str(relative)
            temporary.replace(target)
            copied += 1
        assert target.stat().st_size == item['bytes'], str(relative)
        assert digest(target) == item['sha256'], str(relative)
    summary = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        integrity_manifest_sha256=digest(manifest), recordings=len(report['files']),
        sidecars=len(report['sidecar_files']), copied_this_pass=copied,
        all_native_hashes_and_sizes_match=True, partials=len(report.get('partial_files', [])))
    verification.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--study', choices=('workspace','layout','layout-extra','wide'), default='workspace')
    args = p.parse_args()
    assert args.manifest.resolve().is_relative_to(PROJECT)
    assert args.verification.resolve().is_relative_to(PROJECT)
    main(args.manifest, args.verification, args.study)
