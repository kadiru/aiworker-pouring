"""CPU-only clearance screen for an opt-in late-pour lip translation.

Candidate files live separately from the running simulator. This checks fixed
geometry and sampled IK poses, not droplet trajectories or task success.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
p = argparse.ArgumentParser()
p.add_argument('--candidate-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--shift', type=float, default=.008)
a = p.parse_args()
assert a.candidate_dir.is_absolute() and not a.output.exists()
assert a.shift == .008
sys.path[:0] = [str(a.candidate_dir), '/workspace/src',
               '/workspace/learning/A-finish-20261106-v1/staged', '/workspace']
from check_workspace_clearance import geometry, intersections
import pour_wide_motion
from pour_wide_motion import project_target, centered_return_path, late_mouth_offset
assert Path(pour_wide_motion.__file__).resolve() == a.candidate_dir / 'pour_wide_motion.py'
from preflight_finish_study import vessel, object_hits
from robot_pour import build_path
from pour_finish_study import plan, ROOT
from bottle_pour import BOTTLE, CUP, SIDES
from pour_wide_geometry import calibrated_pose
from wide_gripper_geometry import finger_meshes, REST_ERRORS
import check_workspace_clearance, preflight_finish_study, wide_gripper_geometry
import numpy as np
import torch
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


stage, _, meshes = geometry()
assert len(meshes) == 8
for angles in itertools.product((.64, .75), (.75, .95), (.64, .75), (.75, .95)):
    meshes.extend(finger_meshes(stage, angles))
model = json.loads((ROOT / 'model-summary.json').read_text())
assert model['checkpoint_sha256'] == 'bc9dc0e2d0afd2d090ff36f283be2276a9090f5fc48d3bf8290291dfcf2ee81e'
sources = dict(model['runtime_sources'])
for name, digest in sources.items():
    assert sha(Path('/workspace/src') / name) == digest, name
for name in ('pour_wide_motion.py', 'pour_guard_actor.py'):
    sources[name] = sha(a.candidate_dir / name)

# The translated bottle is entirely above the cup rim during the active ramp.
# Include its 2.4 mm shell and an additional 4.4 mm collision allowance.
outer = np.array([[radius + .0024, angle, z] for z, radius in BOTTLE
                  for angle in np.arange(40) * 2 * np.pi / 40])
local = np.c_[outer[:, 0] * np.cos(outer[:, 1]), outer[:, 0] * np.sin(outer[:, 1]), outer[:, 2]]
minimum_vertical_margin = float('inf')
for phase in np.linspace(116., 165., 197):
    base, rot = calibrated_pose(phase, 1., [0., 0.], .052)
    base[0] += late_mouth_offset(phase, a.shift)
    margin = float((local @ rot.T + base)[:, 2].min() - CUP[-1][0] - .0044)
    minimum_vertical_margin = min(minimum_vertical_margin, margin)
assert minimum_vertical_margin > 0.

# Conservative inscribed-circle margin of the projected inner lip at the
# observed failing angle. This is geometry only, not a prediction of fluid.
lip_margins = {}
for shift in (0., a.shift):
    phase = np.radians(125.43763575786772)
    angles = np.linspace(0., 2*np.pi, 721)
    xy = np.c_[-.013 + shift + .009*np.cos(angles)*np.cos(phase), .009*np.sin(angles)]
    lip_margins[str(shift)] = float(CUP[-1][1]*np.cos(np.pi/SIDES) - np.linalg.norm(xy, axis=1).max())
assert lip_margins[str(a.shift)] > lip_margins['0.0'] > 0.

rows = []
report = dict(candidate_runtime_sources=sources, parent_checkpoint_sha256=model['checkpoint_sha256'],
              late_mouth_shift_m=a.shift, shift_angle_range_deg=[116., 122.],
              minimum_carried_bottle_vertical_collision_margin_m=minimum_vertical_margin,
              projected_inner_lip_inscribed_circle_margins_at_failure_m=lip_margins,
              hulls=len(meshes), closed_grip_ranges_rad=[[.64,.75],[.75,.95],[.64,.75],[.75,.95]],
              rest_transform_discrepancies=REST_ERRORS, cases=rows, complete=False,
              helper_sha256=sha(Path(__file__)),
              geometry_helper_sources={module.__name__: dict(path=module.__file__, sha256=sha(Path(module.__file__)))
                  for module in (check_workspace_clearance, preflight_finish_study, wide_gripper_geometry)},
              scope='Authored fixtures, sampled IK and closed-gripper convex hulls. No simulation, fluid, self-collision or task-success claim. Candidate sources are separate from the active runtime.')
for spec in plan()['development'] + plan()['test']:
    _, lift, tilt, arm = build_path(stage, spec, include_planner=True)
    cup = np.array(spec['cup_world'])
    source = cup + [-spec['source_xy'][1], spec['source_xy'][0], 0.]
    row = dict(name=spec['name'], late_pour=[], returns=[], ready=True)
    q = np.radians(tilt[36])
    for phase in np.arange(110., 130.0001, .5):
        q = project_target(arm, q, source[:2], cup[:2], cup[2], phase, spec['mouth_z'], late_mouth_shift_m=a.shift)
        bench = intersections(arm, meshes, q)
        contact = object_hits(arm, meshes, q, vessel(cup, 'cup'))
        row['late_pour'].append(dict(phase_deg=float(phase), bench=bench, cup=contact))
        row['ready'] &= not bench and not contact
    for phase in (127., 165.):
        start = project_target(arm, np.radians(tilt[round(phase/3)]), source[:2], cup[:2], cup[2], phase, spec['mouth_z'], late_mouth_shift_m=a.shift)
        times, path = centered_return_path(arm, start, phase, source[:2], cup[:2], cup[2], np.radians(lift[-1]), spec['mouth_z'], drain_hold_s=0., late_mouth_shift_m=a.shift)
        r = dict(start_phase_deg=phase, duration_s=float(times[-1]), poses=len(path), bench=[], cup=[])
        assert abs(times[-1] - 3.1) < 1e-8
        for t, q in zip(times, path):
            for key, hits in (('bench', intersections(arm, meshes, q)), ('cup', object_hits(arm, meshes, q, vessel(cup, 'cup')))):
                if hits:
                    r[key].append(dict(t=float(t), hits=hits))
        row['ready'] &= not r['bench'] and not r['cup']
        row['returns'].append(r)
    rows.append(row)
    temporary = a.output.with_suffix('.partial.json')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(a.output)
    print(json.dumps(dict(name=row['name'], ready=row['ready'], cases=len(rows))), flush=True)
assert len(rows) == 38 and all(r['ready'] for r in rows)
assert not torch.cuda.is_initialized()
report.update(complete=True, cuda_initialized=False)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print('LATE_DRIP_GEOMETRY_SCREEN_PASSED', flush=True)
