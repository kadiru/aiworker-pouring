"""Read existing fit/label reports by declared cup size; no inference or GPU use."""
import hashlib
import json
from pathlib import Path

from perception_fit_data import admitted_scenes, admitted_tasks

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / 'evidence/perception-A'


def main():
    files = {
        'plan': EVIDENCE / 'plan-r3-exact-quantiles.json',
        'locator': EVIDENCE / 'cup-neural-localization-task-r3.json',
        'selection': EVIDENCE / 'scope-selection-task-r3.json',
        'audit': EVIDENCE / 'development-r3-audit.json',
    }
    data = {k: json.loads(p.read_text()) for k, p in files.items()}
    plan, locator = data['plan'], data['locator']
    static = admitted_scenes(plan)
    admitted = {s['name']: (s, split) for s, split, _ in static + admitted_tasks(plan)}
    records = {r['name']: r for r in data['audit']['records']}
    rows = []
    for split, key in [('train', 'training_check'), ('validation', 'selected')]:
        for row in locator[key]['rows']:
            if row['domain'] != 'native_detail':
                continue
            spec, assigned = admitted[row['name']]
            assert assigned == split
            size = (spec.get('cup_height_scale', 1.), spec.get('cup_diameter_scale', 1.))
            assert row['baseline_cup'] == (size == (1., 1.))
            rows.append(dict(row, split=split, size=size))
    limit = plan['component_gates']['localization_max_error_mm']
    threshold = data['selection']['threshold']
    groups = []
    for size in sorted({r['size'] for r in rows}):
        group = dict(height_scale=size[0], diameter_scale=size[1], splits={})
        for split in ('train', 'validation'):
            sample = [r for r in rows if r['size'] == size and r['split'] == split]
            assert sample
            labels = []
            for spec, assigned, _ in static:
                if assigned != split or (spec.get('cup_height_scale', 1.), spec.get('cup_diameter_scale', 1.)) != size:
                    continue
                label = records[spec['name']]['label']
                assert label['accepted']
                labels.append(dict(name=spec['name'], stable_height=label['target']))
            worst = max(sample, key=lambda r: r['error_mm'])
            group['splits'][split] = dict(
                frames=len(sample), episodes=len({r['name'] for r in sample}),
                frames_above_localization_limit=sum(r['error_mm'] >= limit for r in sample),
                frames_below_positive_score_threshold=sum(r['score'] < threshold for r in sample),
                maximum_error_mm=worst['error_mm'],
                worst_frame={k: worst[k] for k in ('name', 'group', 'index', 'score')},
                stationary_label_episodes=len(labels),
                stationary_near_full_episodes=sum(r['stable_height'] > .8 for r in labels),
                stationary_label_range=[min(r['stable_height'] for r in labels), max(r['stable_height'] for r in labels)] if labels else None,
                stationary_labels=labels,
            )
        groups.append(group)
    assert max(g['splits']['validation']['maximum_error_mm'] for g in groups) == data['selection']['all_size_max_validation_error_mm']
    report = dict(
        scope='Read-only aggregation of existing round-3 fit/validation reports and admitted stationary labels. No new images, inference, training, reserved-test observations or runtime changes. Sizes are declared fixture dimensions. Frame errors do not establish temporal confirmation or autonomous success.',
        localization_limit_mm=limit, positive_score_threshold=threshold,
        near_full_label_threshold=.8,
        groups=groups,
        resized_autonomy_accepted=False,
        sources={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    out = PROJECT / 'evidence/size-A/perception-readiness-r3.json'
    assert not out.exists(), 'Preserve the recorded readiness snapshot'
    out.write_text(json.dumps(report, indent=2) + '\n')
    for group in groups:
        print(json.dumps({k: v for k, v in group.items() if k != 'splits'} | {
            split: {k: values[k] for k in ('frames', 'maximum_error_mm', 'stationary_near_full_episodes')}
            for split, values in group['splits'].items()}))


if __name__ == '__main__':
    main()
