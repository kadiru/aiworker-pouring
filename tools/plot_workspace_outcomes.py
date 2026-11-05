"""Plot actual task scores at specified start poses and measured terminal levels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PASS, FAIL = '#176795', '#B84232'
FULL = 'cup_near_full'


def point(ax, x, y, row):
    passed = row['surface_goal_zero_spill_success']
    ax.scatter(x, y, c=PASS if passed else FAIL,
        marker='o' if row['expected_stop'] == FULL else '^',
        s=55, edgecolors='white', linewidths=.7, zorder=4)


def main(report, output, development):
    r = json.loads(report.read_text())
    rows = r['development'] if development else r['cases']
    if not development:
        assert r['status'] == 'complete' and len(rows) == 30
        assert r['independently_recomputed_primary_scores'] == 30
    assert rows and all(x['status'] == 'complete' for x in rows)
    successes = sum(x['surface_goal_zero_spill_success'] for x in rows)
    contained = sum(x['zero_spill'] for x in rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.9),
        gridspec_kw={'height_ratios': [1.45, 1.]})
    fig.subplots_adjust(left=.10, right=.96, top=.82, bottom=.155, hspace=.54, wspace=.32)
    fig.text(.10, .967, 'AIWorker pouring across wider placements',
        fontsize=19, fontweight='bold', color='#172636', va='top')
    rounds = {x.get('development_round') for x in rows}
    kind = ('Development r' + str(next(iter(rounds))) if development and len(rounds)==1
        else 'Development' if development else 'Held-out simulation')
    fig.text(.10, .922, f'{kind}  |  {successes}/{len(rows)} strict passes  |  '
        f'{contained}/{len(rows)} with zero measured loss', fontsize=12, color='#35495C')
    for ax, kind, old, expanded, title in zip(axes[0], ('cup', 'bottle'),
            ((8, 8), (12, 17)), ((24, 24), (36, 51)), ('Cup start positions', 'Bottle start positions')):
        ax.add_patch(Rectangle((-old[0], -old[1]), 2*old[0], 2*old[1],
            facecolor='#E6E9EC', edgecolor='#758393', linestyle='--', linewidth=1))
        ax.add_patch(Rectangle((-expanded[0], -expanded[1]), 2*expanded[0], 2*expanded[1],
            fill=False, edgecolor='#8997A4', linewidth=1))
        for row in rows:
            cx, cy, _ = row['cup_world']; sx, sy = row['source_xy']
            x, y = (cx-.49, cy+.15) if kind == 'cup' else (cx-sy-.49, cy+sx+.27)
            point(ax, 1000*x, 1000*y, row)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.15*expanded[0], 1.15*expanded[0]); ax.set_ylim(-1.15*expanded[1], 1.15*expanded[1])
        ax.set_title(title, loc='left', fontweight='bold'); ax.set_xlabel('X offset (mm)'); ax.set_ylabel('Y offset (mm)')
    for ax, target in zip(axes[1], (FULL, 'bottle_empty')):
        group = [(i, row) for i, row in enumerate(rows) if row['expected_stop'] == target]
        ax.set_xlabel('Case index in this batch'); ax.set_xlim(-.8, len(rows)-.2)
        ticks = list(range(0, len(rows), 1 if len(rows)<10 else 5))
        ax.set_xticks(ticks if ticks[-1] == len(rows)-1 else ticks+[len(rows)-1])
        if target == FULL:
            ax.set_title('Full-cup cases', loc='left', fontweight='bold'); ax.set_ylabel('Final rim height (%)')
            ax.axhline(90, linestyle='--', color='#556777', linewidth=1)
            vals = [100*x['cup_surface_height_fraction'] for _, x in group]
            ax.set_ylim(max(0, min([87]+vals)-1), max([98]+vals)+1)
        else:
            ax.set_title('Empty-bottle cases', loc='left', fontweight='bold'); ax.set_ylabel('Liquid left (nominal mL)')
            vals = [x['source_remaining_ml'] for _, x in group]
            limits = [max(.25, .01*x['source_initial_ml']) for _, x in group]
            for (i, _), lim in zip(group, limits): ax.hlines(lim, i-.45, i+.45, colors='#556777', linewidth=1.5)
            ymax = max([.4]+vals+limits)
            if ymax > 5 * max([.4]+limits):
                # A bottle never picked up can still contain tens of mL.
                # Keep its true value while making sub-mL residuals and limits legible.
                ax.set_yscale('symlog', linthresh=.5)
                ax.set_ylim(-.065, 1.45*ymax)
                ticks = [v for v in (0, .25, .5, 1, 5, 20, 40, 100) if v <= 1.4*ymax]
                ax.set_yticks(ticks, labels=[f'{v:g}' for v in ticks])
                ax.set_ylabel('Liquid left (nominal mL)\nlog scale above 0.5 mL')
            else:
                ax.set_ylim(-.04*ymax, 1.12*ymax)
        for (i, row), value in zip(group, vals): point(ax, i, value, row)
    for ax in axes.flat:
        ax.set_axisbelow(True); ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9); ax.title.set_fontsize(11)
    legend = [Line2D([], [], marker=m, color='none', markerfacecolor=c, markeredgecolor=c,
                    markersize=6, label=label) for m,c,label in
        [('o', '#596978', 'Full-cup goal'), ('^', '#596978', 'Empty-bottle goal'),
         ('s', PASS, 'Strict pass'), ('s', FAIL, 'Strict failure')]]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.09,.902),
        ncol=4, frameon=False, fontsize=9, columnspacing=1.45, handletextpad=.35)
    fig.text(.10, .048, 'Start-pose panels: grey area = previous range; outline = expanded range. '
        'Level panels: lines mark the full/empty criteria.\nZero measured loss = zero final outside liquid '
        'and zero sampled near-table loss at 20 Hz. Nominal mL are particle-count units.',
        fontsize=8.3, color='#465866', linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); fig.savefig(output.with_suffix('.pdf')); plt.close(fig)
    print(output)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--development', action='store_true')
    a=p.parse_args(); main(a.report, a.output, a.development)
