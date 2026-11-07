"""Archive the idle r2 candidate and activate a separately checked level refit.

Must be followed by guarded deployment/restart of the two staged runtime
files. This helper neither restarts nor queues anything.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import urllib.request

from calibrate_finish_target import ROOT, OUTPUTS, read, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--candidate-sources', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--withdrawal', type=Path, required=True)
    a = p.parse_args()
    assert a.candidate.parent == ROOT
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(OUTPUTS.glob('*/*FN30_A_test*/parameters.json'))
    with urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=10) as response:
        h = json.load(response)
    assert not h['queue'] and not (h.get('recording') or {}).get('active')
    assert not h.get('error') and not h.get('learning_error')
    old = read(ROOT / 'model-summary.json')
    previous = sha(ROOT / 'best.pt')
    assert previous == old['checkpoint_sha256'] == 'bc9dc0e2d0afd2d090ff36f283be2276a9090f5fc48d3bf8290291dfcf2ee81e'
    new = read(a.candidate / 'model-summary.json')
    refit = read(a.candidate / 'level-refit.json')
    assert new['level_refit'] == refit
    assert refit['checkpoint_sha256'] == new['checkpoint_sha256'] == sha(a.candidate / 'best.pt')
    assert refit['parent_checkpoint_sha256'] == previous and new['status'] == 'fit_complete'
    assert refit['source_helper_sha256'] == sha(Path(__file__).with_name('refit_finish_level.py'))
    assert refit['review_sha256'] == sha(a.review) and refit['withdrawal_sha256'] == sha(a.withdrawal)
    assert refit['integrity_sha256'] == sha(ROOT / 'development-recording-integrity.json')
    assert new['data_manifest_sha256'] == sha(a.candidate / 'data-manifest.json')
    assert new['scene_plan_sha256'] == sha(ROOT / 'frozen-plan.json')
    assert '4070' in new['device'] and new['terminal_drain_hold_s'] == 0.
    assert new['late_mouth_shift_m'] == .008
    assert new['thresholds']['coarse_brake_target'] == .82 and new['thresholds']['max_pauses'] == 13
    assert new['thresholds']['full_settled_target'] == refit['new_full_target'] <= .94
    assert new['late_drip_alignment']['geometry_sha256'] == sha(ROOT / 'late-drip-clearance-r3.json')
    for n, digest in old['runtime_sources'].items():
        assert sha(Path('/workspace/src') / n) == digest, n
    manifest = read(a.source_manifest)
    assert refit['source_manifest_sha256'] == sha(a.source_manifest)
    assert manifest['files'] == new['runtime_sources'] == new['sources']
    changed = {n for n,h in manifest['files'].items() if h != old['runtime_sources'][n]}
    assert changed == {'pour_guard_actor.py', 'pour_wide_motion.py'}
    for n in changed:
        assert sha(a.candidate_sources / n) == manifest['files'][n]
    archive = ROOT / 'candidates/v23-r2'
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'src').mkdir()
    for n in old['runtime_sources']:
        shutil.copy2(Path('/workspace/src') / n, archive / 'src' / n)
    for n in ('best.pt','model-summary.json','data-manifest.json','frozen-plan.json','calibration-anchor-manifest.json',
              'development-recording-integrity.json','recording-repair.json','infrastructure-aborts.json',
              'development-resume-plan-r0.json','full-stop-calibration.json','initial-pour-calibration.json',
              'coarse-brake-window-probe.json','late-drip-clearance-r3.json'):
        shutil.copy2(ROOT / n, archive / n)
    shutil.copy2(a.review, archive / 'development-review.json')
    shutil.copy2(a.withdrawal, archive / 'development-withdrawal.json')
    for origin, name in (('development-r2-case03-spill-diagnostic.json','late-loss-diagnostic-r2.json'),
                         ('development-r2-case05-full-stop-diagnostic.json','full-stop-diagnostic-r2.json')):
        shutil.copy2(a.candidate_sources / origin, archive / name)
        shutil.copy2(a.candidate_sources / origin, ROOT / name)
    for n in changed:
        shutil.copy2(a.candidate_sources / n, ROOT / 'staged' / n)
    for n in ('data-manifest.json','calibration-anchor-manifest.json','level-refit.json'):
        shutil.copy2(a.candidate / n, ROOT / n)
    release = copy.deepcopy(manifest)
    release.update(checkpoint_sha256=new['checkpoint_sha256'], fitted_sources=new['sources'],
        fit_source_commit=manifest['source_commit'], fitting_helper_source_commit=refit['source_helper_commit'],
        changed_fitted_source_files=[], level_refit_sha256=sha(ROOT / 'level-refit.json'),
        parent_fitted_sources=old['sources'],
        scope='Supervised settled-level refit with explicitly consumed r0-r2 development images, plus late-pour lip alignment. Other learned components, physics, natural return timing and scoring unchanged. Eight r3 native development attempts required before selection.')
    (ROOT / 'runtime-source-manifest-drip-r3.json').write_text(json.dumps(release, indent=2)+'\n')
    shutil.copy2(a.candidate / 'best.pt', ROOT / 'best.refit.partial.pt')
    shutil.copy2(a.candidate / 'model-summary.json', ROOT / 'model-summary.refit.partial.json')
    (ROOT / 'best.refit.partial.pt').replace(ROOT / 'best.pt')
    (ROOT / 'model-summary.refit.partial.json').replace(ROOT / 'model-summary.json')
    print(json.dumps(dict(checkpoint_sha256=new['checkpoint_sha256'], archived_parent=previous,
        next_action='Deploy the two staged runtime sources, restart only the owned container, verify isolation and source identity, then explicitly queue development r3.')))


if __name__ == '__main__':
    main()
