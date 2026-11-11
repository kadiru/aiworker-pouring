"""Test camera transforms, then fit a separate candidate without reserved tests."""
import json
import os
import subprocess
import sys
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    assert not (ROOT/'component-selection.json').exists()
    if (ROOT/'physical-crop-contracts.json').exists():
        report = read(ROOT/'physical-crop-contracts.json')
        assert report['module_sha256'] == sha(ROOT/'physical_level_augmentation.py')
        assert report['test_sha256'] == sha(ROOT/'test_physical_level_augmentation.py')
    else:
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_physical_level_augmentation'],
                                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                capture_output=True, text=True)
        report = dict(passed=result.returncode == 0, returncode=result.returncode,
                      output=result.stdout+result.stderr,
                      module_sha256=sha(ROOT/'physical_level_augmentation.py'),
                      test_sha256=sha(ROOT/'test_physical_level_augmentation.py'))
        with (ROOT/'physical-crop-contracts.json').open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report), flush=True)
    assert report['passed']
    for name in ('prepare_physical_crop_levels.py', 'fit_physical_crop_levels.py'):
        subprocess.run([sys.executable, '-u', str(ROOT/name)], check=True)
    protected_unchanged()
    report = dict(passed=read(ROOT/'pair-r11.json')['passed'],
                  pair_report_sha256=sha(ROOT/'pair-r11.json'), reserved_started=False)
    with (ROOT/'physical-crop-development.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
