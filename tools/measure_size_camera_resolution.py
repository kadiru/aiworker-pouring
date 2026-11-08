"""Ideal projection diagnostic; fixture geometry is not supplied to the policy."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
sys.path[:0] = ['/workspace/src', '/workspace']
from pour_guard_vision import project
from pour_vessel_geometry import cup_geometry
from pour_size_study import ROOT


def main():
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    plan = json.loads((ROOT / 'plan.json').read_text())
    rows = []
    for spec in plan['calibration_seed'] + plan['recovery_development']:
        cal = copy.deepcopy(saved['calibration'])
        geometry = cup_geometry(spec)
        center = np.asarray(spec['cup_world'])
        bottom, rim = geometry['bottom_z_m'], geometry['rim_z_m']
        low = project(center + [0, 0, bottom], cal)
        high = project(center + [0, 0, rim], cal)
        right = np.asarray(cal['T'])[:3, 0]
        outer = project(center + [0, 0, rim] + right * geometry['rim_radius_m'], cal)
        p90 = project(center + [0, 0, bottom + .90 * (rim - bottom)], cal)
        p91 = project(center + [0, 0, bottom + .91 * (rim - bottom)], cal)
        rows.append(dict(name=spec['name'], cup_geometry=geometry,
                         projected_inner_height_px=float(np.linalg.norm(high - low)),
                         projected_vertical_height_px=float(abs(high[1] - low[1])),
                         projected_rim_width_px=float(2 * abs(outer[0] - high[0])),
                         one_percentage_point_near_full_px=float(np.linalg.norm(p91 - p90))))
    result = dict(scope='Ideal 384x384 task-camera projection at declared fixture positions. No new rendering, fitting, or controller inputs. This describes pixel scale, not a proven localization or level-error bound.',
                  checkpoint_sha256=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest(),
                  cases=rows, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'camera-resolution.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows), inner_height_range_px=[min(r['projected_inner_height_px'] for r in rows), max(r['projected_inner_height_px'] for r in rows)],
                          one_percentage_point_range_px=[min(r['one_percentage_point_near_full_px'] for r in rows), max(r['one_percentage_point_near_full_px'] for r in rows)])))


if __name__ == '__main__':
    main()
