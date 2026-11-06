"""Native CPU geometry screening before any new wet reference or reserved test.

Includes the four open finger collision hulls. This checks initial fixtures,
bench and cup clearance along the approach; it is not a dynamics guarantee.
"""
import sys
sys.path.insert(0, '/workspace')
from check_workspace_clearance import geometry, intersections
import hashlib
import json
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdPhysics
from robot_pour import build_path, interpolate, ROOT as ROBOT
from pour_guard_vision import project
from pour_wide_study import ROOT, PREFIX, REFERENCE_PREFIX, candidates, TASK_FOCAL_M


def vessel(center, kind, margin=.002):
    profile = [(.004, .025), (.082, .025), (.107, .009), (.135, .009)] if kind == 'bottle' else [(.004, .020), (.038, .024)]
    # Include the actual 2.4mm shell and two 1mm contact offsets. The first
    # preflight used the inner fluid profile and missed a near-tangent contact.
    points = np.array([[(r+.0024+margin)*np.cos(a), (r+.0024+margin)*np.sin(a), z+dz]
                      for z, r in profile for dz in (-margin,margin)
                      for a in np.arange(24)*2*np.pi/24]) + center
    return points, ConvexHull(points).equations


def transforms(planner, q):
    xyz, rot, result = planner.shoulder.copy(), np.eye(3), []
    for delta, axis, angle in zip(planner.offsets, planner.axes, q):
        xyz = xyz + rot @ delta
        rot = rot @ Rotation.from_rotvec(axis*angle).as_matrix()
        result.append((xyz.copy(), rot.copy()))
    return result


def object_hits(planner, meshes, q, shape):
    points, eqv = shape
    poses = transforms(planner, q)
    hits = []
    for i, name, local, eq in meshes:
        xyz, rot = poses[i]
        mp = local @ rot.T + xyz
        if not (np.minimum(mp.max(0), points.max(0)) > np.maximum(mp.min(0), points.min(0)) + 1e-7).all():
            continue
        normal = eq[:, :3] @ rot.T
        result = linprog(np.zeros(3), A_ub=np.vstack((normal, eqv[:, :3])),
            b_ub=np.r_[-eq[:, 3] + normal @ xyz, -eqv[:, 3]], bounds=[(None, None)]*3, method='highs')
        assert result.status in (0, 2), result.message
        if result.success:
            hits.append(name)
    return hits


def main():
    assert not (ROOT / 'frozen-plan.json').exists()
    stage, planner, meshes = geometry()
    cache = UsdGeom.XformCache()
    inv = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT + '/right_arm/arm_r_link7'))).T)
    for prim in stage.Traverse():
        name = str(prim.GetPath())
        if '/right_arm/gripper_r_' not in name or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        m = inv @ np.array(cache.GetLocalToWorldTransform(prim)).T
        local = points @ m[:3, :3].T + m[:3, 3]
        hull = ConvexHull(local)
        meshes.append((6, name, local[hull.vertices], hull.equations))
    assert len(meshes) == 12
    base = json.loads(open('/workspace/learning/A-layout-variety-20261105-v1/frozen-plan.json').read())['test'][0]
    import torch
    cal = torch.load('/workspace/learning/A-workspace-20261104-v1/best.pt', map_location='cpu', weights_only=False)['calibration']
    cal['K'] = np.asarray(cal['K']).copy()
    cal['K'][0, 0] *= TASK_FOCAL_M / .040
    cal['K'][1, 1] *= TASK_FOCAL_M / .040
    report, accepted, all_scenes = [], {}, set()
    for split, pairs, seed in [('train', 12, 202611051), ('validation', 4, 202611052),
                               ('development', 4, 202611053), ('test', 15, 202611054)]:
        chosen = []
        for pair in candidates(base, split, seed):
            checked = []
            for s in pair:
                cup = np.array(s['cup_world'])
                bottle = cup + [-s['source_xy'][1], s['source_xy'][0], 0.]
                record = dict(split=split, candidate_index=s['candidate_index'], swapped=s['swapped'],
                              cup=cup.tolist(), bottle=bottle.tolist(), ready=False)
                try:
                    approach, lift, tilt = build_path(stage, s)
                    approach = np.radians(approach)
                    home = approach[0]
                    initial = {kind: object_hits(planner, meshes, home, vessel(center, kind))
                               for kind, center in [('cup', cup), ('bottle', bottle)]}
                    record['initial_hits'] = initial
                    if any(initial.values()):
                        checked.append(record)
                        continue
                    bench, cups = [], []
                    cup_shape = vessel(cup, 'cup')
                    for u in np.linspace(0., 1., 41):
                        q = interpolate(approach, u)
                        if intersections(planner, meshes, q):
                            bench.append(float(u))
                        if object_hits(planner, meshes, q, cup_shape):
                            cups.append(float(u))
                    anchors = [project(bottle + [0, 0, .135], cal), project(cup + [0, 0, .038], cal)]
                    visible = all(24 <= uv[0] <= 360 and 24 <= uv[1] <= 350 for uv in anchors)
                    record.update(bench_intersections=bench, cup_approach_intersections=cups,
                                  initial_anchor_uv=[uv.tolist() for uv in anchors], anchors_inside_image=visible,
                                  ready=not bench and not cups and visible)
                except ValueError as error:
                    record['ik_error'] = str(error)
                checked.append(record)
            report.extend(checked)
            if all(r['ready'] for r in checked):
                pair_index = len(chosen) // 2
                for s, check in zip(pair, checked):
                    i = len(chosen)
                    suffix = ('demo_' + split) if split in ('train', 'validation') else ('dev' if split == 'development' else 'test')
                    prefix=REFERENCE_PREFIX if split in ('train','validation') else PREFIX
                    s['name'] = f'{prefix}{suffix}_{i:02}'
                    s['pair_index'] = pair_index
                    check['accepted_name'] = s['name']
                    identity = tuple(s['cup_world'] + s['source_xy'])
                    assert identity not in all_scenes
                    all_scenes.add(identity)
                    chosen.append(s)
                print(json.dumps(dict(split=split, accepted=len(chosen), candidate=pair[0]['candidate_index'])), flush=True)
            (ROOT / 'preflight-progress.json').write_text(json.dumps(dict(meshes=12, checked=report), indent=2)+'\n')
            if len(chosen) == pairs*2:
                break
        assert len(chosen) == pairs*2, split
        accepted[split] = chosen
    result = dict(demonstrations=accepted['train']+accepted['validation'], development=accepted['development'], test=accepted['test'],
        test_seed=202611054, test_pairs=15, test_full=15, test_empty=15, task_camera_focal_m=TASK_FOCAL_M,
        scope='Five wide spatial families, jittered placements and exact swaps. Geometry-only screening before physics; no test images or outcomes used for fitting or selection.',
        geometry_scope='Twelve arm/palm/open-finger convex hulls; initial bottle/cup overlap; bench and cup clearance at 41 approach poses; reference full-pour IK; fixed-camera anchor visibility. Excludes native dynamics and all-angle self-collision.',
        attempts_per_test=1, test_rule='Freeze final candidate before thirty once-only reserved tests. Retain every outcome; no fitting or retries during tests.')
    (ROOT / 'preflight.json').write_text(json.dumps(dict(meshes=12, checked=report,
        selected_count=sum(len(v) for v in accepted.values()), rejected_candidates=sum(not r['ready'] for r in report)), indent=2)+'\n')
    (ROOT / 'frozen-plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print('FROZEN_PLAN ' + hashlib.sha256((ROOT / 'frozen-plan.json').read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
