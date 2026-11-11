"""Native-image augmentation and bounded fit, with no reserved-test side effects."""
import json
import subprocess
import sys
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    assert not read(ROOT/'pair-r12.json')['passed']
    for name in ('cache_native_shifted_levels.py', 'prepare_native_crop_levels.py', 'fit_native_crop_levels.py'):
        print(json.dumps(dict(starting=name)), flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT/name)], check=True)
    protected_unchanged()
    report = dict(passed=read(ROOT/'pair-r13.json')['passed'],
                  pair_report_sha256=sha(ROOT/'pair-r13.json'), reserved_started=False)
    with (ROOT/'native-crop-development.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
