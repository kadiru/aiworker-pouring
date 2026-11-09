"""Archive closed reserved outcomes and refresh status; never queue or tune."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/side-return-A'


def main():
    result = subprocess.run(
        ['ssh', 'kfuUbuntu', 'python3 /tmp/snapshot_side_progress.py --part test'],
        check=True, capture_output=True, text=True, timeout=60,
    )
    report = json.loads(result.stdout)
    frozen = json.loads((EVIDENCE / 'final-selection.json').read_text())
    assert report['part'] == 'test' and report['planned'] == 30
    for key in ('checkpoint_sha256', 'plan_sha256'):
        assert report[key] == frozen[key]
    assert report['helper_sha256'] == hashlib.sha256(
        (ROOT / 'tools/snapshot_side_progress.py').read_bytes()).hexdigest()
    count = report['completed']
    assert 0 < count <= 30 and len(report['cases']) == count
    assert len({c['name'] for c in report['cases']}) == count
    assert report['accepted'] == sum(c['accepted'] for c in report['cases'])
    destination = EVIDENCE / f'test-r1-progress-after{count:02d}.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == report
    else:
        with destination.open('x') as output:
            output.write(json.dumps(report, indent=2) + '\n')
    zero = sum(c['zero_spill'] for c in report['cases'])
    summary = (f"Reserved progress: {report['accepted']}/{count} completed attempts accepted; "
               f"{zero}/{count} have zero final/table loss; {30-count} remain.")
    status = ROOT / 'STATUS.md'
    previous = status.read_text()
    updated, matches = re.subn(r'Reserved progress: [^.]*\.', summary, previous, count=1)
    assert matches == 1
    if updated != previous:
        status.write_text(updated)
    last = report['cases'][-1]
    placement_error = last['side'].get('final_placement_error_m')
    print(json.dumps(dict(completed=count, accepted=report['accepted'], zero_loss=zero,
        latest=last['name'], latest_accepted=last['accepted'],
        latest_failures=last['acceptance_failures'], height=last['height_fraction'],
        source_ml=last['source_ml'], side=last['side'].get('side'),
        placement_error_mm=None if placement_error is None else 1000*placement_error)))


if __name__ == '__main__':
    main()
