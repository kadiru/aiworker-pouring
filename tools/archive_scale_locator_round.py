"""Preserve a rejected locator round and declare the next controlled refinement."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/tmp')
import watch_wide_collection as w

ROOT = w.HOST_ROOT / 'learning/A-scale-20261109-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--round',choices=('r0','r1'),default='r0')
    args=parser.parse_args();round_name=args.round;next_round='r1' if round_name=='r0' else 'r2'
    manifest = read(ROOT / 'candidate.json')
    plan = read(ROOT / 'plan.json')
    witness = w.isolation()
    health = w.health()
    assert not health['queue'] and not health['recording']['active'] and not health.get('error')
    assert not manifest['trained'] and not (ROOT / 'component-selection.json').exists()
    assert sha(ROOT / 'best.pt') == manifest['checkpoint_sha256']
    assert sha(ROOT / 'plan.json') == manifest['plan_sha256']
    assert all(sha(w.HOST_ROOT / 'src' / n) == d for n, d in manifest['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT / Path(p).relative_to('/workspace')) == d
               for p, d in manifest['protected_parent_files'].items())
    started = {read(p)['name'] for p in (w.HOST_ROOT / 'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    report = read(ROOT / 'cup-neural-localization.json')
    completion = read(ROOT / ('locator-fit-'+round_name) / 'completion.json')
    assert report['passed'] is False and completion['exit_code'] == 0
    assert completion['report_sha256'] == sha(ROOT / 'cup-neural-localization.json')
    assert sha(ROOT / 'fit_scale_locator.py') == report['source_sha256']
    assert not (ROOT / 'cup-neural-localization.pt').exists()
    assert not (ROOT / 'scope-selection.json').exists()
    archive = ROOT / 'locator-rounds' / round_name
    archive.mkdir(parents=True)
    names = ('cup-neural-localization.json', 'cup-neural-localization-attempt.pt',
             'cup-locator-data-manifest.json', 'fit_scale_locator.py',
             'run_scale_locator_fit.py', 'scale_data_admission.py')
    if (ROOT/'scale_locator_augmentation.py').exists():names+=('scale_locator_augmentation.py',)
    hashes = {}
    for name in names:
        shutil.copyfile(ROOT / name, archive / name)
        assert sha(archive / name) == sha(ROOT / name)
        hashes[name] = sha(archive / name)
    code = """
import hashlib,torch
from pathlib import Path
p=Path('/workspace/learning/A-scale-20261109-v1/locator-rounds/ROUND_NAME')
a=torch.load(p/'cup-neural-localization-attempt.pt',map_location='cpu',weights_only=False)
assert a['accepted'] is False
state=a['state']
identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(state.items()))).hexdigest()
torch.save(dict(bank=dict(kind='cup_heatmap',state=state,identity=identity),accepted=False,
    purpose='Initialization only; rejected validation-selected attempt',source_epoch=a['epoch']),p/'refinement-init.pt')
"""
    code=code.replace('ROUND_NAME',round_name)
    subprocess.run(['docker', 'exec', '-i', '-e', 'CUDA_VISIBLE_DEVICES=', '-e',
                    'PYTHONDONTWRITEBYTECODE=1', w.CID, 'nice', '-n', '10',
                    '/isaac-sim/python.sh', '-'], input=code, text=True, check=True)
    hashes['refinement-init.pt'] = sha(archive / 'refinement-init.pt')
    index = dict(round=round_name, passed=False, files=hashes, runtime=witness,
                 best_validation_max_mm=report['best_attempt']['max_error_mm'],
                 original_log_directory=str(ROOT / ('locator-fit-'+round_name)))
    (archive / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    refinement = dict(round=next_round, parent_round=round_name, archive_index_sha256=sha(archive / 'index.json'),
        initialization=f'locator-rounds/{round_name}/refinement-init.pt', initialization_sha256=hashes['refinement-init.pt'],
        plan_sha256=sha(ROOT / 'plan.json'), audit_sha256=sha(ROOT / 'collect-r0-audit.json'),
        hypothesis='Continuous subpixel translations may reduce fixed pixel-phase and heatmap-cell-boundary errors in low-fill views; this is not an established cause.',
        changes=['Bilinear image translations drawn uniformly from [-28,28] pixels, matching target translation',
                 'Initialize from the preserved rejected r0 best validation state'],
        unchanged=['54 episode splits', 'all-size 3 mm gate', 'absence margin', '240 epochs',
                   'AdamW schedule', 'loss', 'nine unused holdouts', 'active robot checkpoint'],
        model_activated=False)
    if round_name=='r1':
        refinement.update(hypothesis='Using 768 of the native 1024 pixels may improve rim precision; no accuracy is assumed.',
            changes=['768x768 input instead of 384x384, with matching intrinsics and projected targets',
                     'Pixel augmentation, absent masks and consensus radius scaled to preserve their original image extent',
                     'Initialize from the rejected r1 best validation state'],
            input_resolution=768,relative_receptive_field='Unchanged 129 input pixels, hence half the previous scene extent')
    plan_path=ROOT / ('locator-refit-'+next_round+'-plan.json')
    with plan_path.open('x') as f:
        json.dump(refinement, f, indent=2)
        f.write('\n')
    for name in names[:3]:
        assert sha(ROOT / name) == hashes[name]
        (ROOT / name).unlink()
    print(json.dumps(dict(archived=True, index_sha256=sha(archive / 'index.json'),
                         refinement_plan_sha256=sha(plan_path))))


if __name__ == '__main__':
    main()
