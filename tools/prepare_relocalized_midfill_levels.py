"""Freeze level refit from the newly replayed image-localization crops."""
import json
from pathlib import Path
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    protected_unchanged()
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    protocol = read(ROOT/'protocol.json'); old = read(ROOT/'protocol-v5-before-midfill.json')
    assert all(protocol[k] == old[k] for k in ('validation', 'holdout', 'gates'))
    tracker = read(ROOT/'tracker-t3.json'); cache = read(ROOT/'level-cache-v6.json')
    assert tracker['passed'] and len(tracker['cases']) == 279
    assert all(r['passed'] for r in tracker['cases']) and all(r['rejected'] for r in tracker['synthetic_challenges'])
    assert cache['passed'] and cache['tracker_sha256'] == sha(ROOT/'tracker-t3.pt')
    assert cache['cache_sha256'] == sha(ROOT/'level-cache-v6.npz') and cache['metadata_sha256'] == sha(ROOT/'level-cache-v6-meta.json')
    assert cache['counts'] == dict(train=dict(episodes=243, frames=3195), validation=dict(episodes=36, frames=918))
    initialization = {}
    files = ['protocol.json', 'protocol-v5-before-midfill.json', 'plan.json', 'candidate.json',
             'tracker-t3.pt', 'tracker-t3.json', 'tracker-t3-protocol.json', 'dynamic_cup_tracker.py',
             'level-cache-v6.npz', 'level-cache-v6-meta.json', 'level-cache-v6.json',
             'perception_midfill_data.py', 'perception_repair_data.py', 'cache_relocalized_midfill.py',
             'fit_relocalized_midfill_levels.py', 'prepare_relocalized_midfill_levels.py',
             'repair_level_model.py', 'repair_level_model_v2.py', 'fit_dynamic_pair.py',
             'scale_level_cnn.py', 'scale_rim_augmentation.py', 'scale_color_mask.py',
             'train-r3-audit.json', 'train-r3-manual-review.json', 'pair-r10.json', 'pair-r11.json']
    for branch in ('rgb', 'color'):
        reports = {rnd: read(ROOT/f'pair-{rnd}-{branch}.json') for rnd in ('r10', 'r11')}
        chosen = min(reports, key=lambda rnd: reports[rnd]['selected']['score'])
        initialization[branch] = f'pair-{chosen}-{branch}.pt'
        assert reports[chosen]['checkpoint_sha256'] == sha(ROOT/initialization[branch])
        files += [f'pair-{rnd}-{branch}.json' for rnd in reports]+[initialization[branch]]
    cfg = dict(round='r12', seed=261110613, locator='tracker-t3.pt',
               protocol_sha256=sha(ROOT/'protocol.json'), frozen_files={n: sha(ROOT/n) for n in files},
               epochs=800, batch_size=64, lr=.00015, lr_steps=[401, 601], jitter_pixels=1.5,
               top_quartile_weight=.25, wall_seconds=1800, max_episode_mae=.02, initialization=initialization,
               method='Independent supervised RGB/color-outline CNNs, geometry-only context; AdamW MSE plus 0.25 worst-quartile MSE, balanced motion/size/episode sampling. All crops rebuilt from tracker-t3; each initialization chosen by the retained development score between r10/r11, never reserved data.',
               scope='243 training / 36 validation episodes. No true-center inference crops, threshold changes, reserved images or active-controller update.')
    with (ROOT/'pair-r12-protocol.json').open('x') as f:
        json.dump(cfg, f, indent=2); f.write('\n')
    print(json.dumps(dict(prepared='r12', initialization=initialization)), flush=True)


if __name__ == '__main__':
    main()
