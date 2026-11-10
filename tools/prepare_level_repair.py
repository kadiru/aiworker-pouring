"""Freeze a bounded independent fit using only previously admitted crops."""
import argparse, hashlib, json
from pathlib import Path
ROOT = Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--round', choices=['r5'], default='r5'); a = p.parse_args()
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    c = read(ROOT/'level-cache-v1.json'); old = read(ROOT/'pair-r4-protocol.json')
    assert c['passed'] and c['source_protocol_sha256'] == sha(ROOT/'pair-r4-protocol.json')
    assert c['cache_sha256'] == sha(ROOT/'level-cache-v1.npz')
    assert c['metadata_sha256'] == sha(ROOT/'level-cache-v1-meta.json')
    assert c['tracker_sha256'] == sha(ROOT/'tracker-t1.pt')
    assert c['tracker_module_sha256'] == sha(ROOT/'dynamic_cup_tracker.py')
    assert all(sha(ROOT/n) == v for n, v in old['frozen_files'].items())
    m = read(ROOT/'candidate.json')
    assert all(sha(Path(n)) == v for n, v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n) == v for n, v in m['runtime_sources'].items())
    files = list(old['frozen_files']) + ['level-cache-v1.npz', 'level-cache-v1-meta.json', 'level-cache-v1.json',
        'cache-level-localization-check.json', 'cache-dynamic-localization-check.json',
        'fit_level_repair.py', 'repair_level_model.py', 'prepare_level_repair.py', 'pair-r4-protocol.json']
    cfg = dict(round=a.round, seed=261110205, locator='tracker-t1.pt', protocol_sha256=sha(ROOT/'protocol.json'),
        frozen_files={n: sha(ROOT/n) for n in sorted(set(files))}, epochs=800, batch_size=64, lr=.0006,
        lr_steps=[401, 601], jitter_pixels=2.5, top_quartile_weight=.25, wall_seconds=1800,
        method='Independent supervised RGB and red-mask/grayscale/edge CNNs, random initialization, equal motion-group/size/episode sampling, MSE plus 0.25 worst-quartile MSE; AdamW. No cross-model loss, no RL.',
        scope='207 existing development episodes only; accepted tracker unchanged; original accuracy gates unchanged; no reserved images or runtime activation.')
    with (ROOT/('pair-'+a.round+'-protocol.json')).open('x') as f: json.dump(cfg, f, indent=2); f.write('\n')
    print(json.dumps(dict(prepared=a.round, epochs_per_branch=800, wall_seconds=1800)))

if __name__ == '__main__': main()
