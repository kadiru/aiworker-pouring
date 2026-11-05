"""CPU-only fixture/IK preflight, using USD collision meshes. No trial execution."""
import json
from pathlib import Path
import sys

# This helper bootstraps the installed USD libraries, and exposes robot geometry.
from check_workspace_clearance import geometry, intersections
import numpy as np
from robot_pour import build_path, interpolate, ArmPath, WRIST
from pour_workspace_path import clearance_approach
from pour_layout_variety import ROOT, PARENT, make_plan


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'selection.json').exists(), 'Cannot revise a frozen evaluation'
    proposed = make_plan(json.loads((PARENT/'frozen-plan.json').read_text())['test'][0])
    stage, hardware, meshes = geometry()
    rows = []
    for spec, layout in zip(proposed['test'], proposed['layouts_world_xy']):
        row = dict(name=spec['name'], label=spec['layout_label'], cup=layout['cup'], bottle=layout['bottle'])
        cup, bottle = np.array(layout['cup']), np.array(layout['bottle'])
        margin = min(*(cup-np.array([.305,-.43])-.024), *(np.array([.685,.03])-cup-.024),
                     *(bottle-np.array([.305,-.43])-.025), *(np.array([.685,.03])-bottle-.025))
        row.update(table_edge_margin_m=float(margin), object_center_separation_m=float(np.linalg.norm(cup-bottle)))
        try:
            approach, lift, tilt = build_path(stage, spec)
            planner = ArmPath(hardware.offsets, ['XYZ'[int(np.argmax(a))] for a in hardware.axes],
                np.degrees(hardware.limits.T), hardware.shoulder)
            planner.q = np.array([-.25,-.45,-.15,-1.4,0.,.25,0.])
            grasp = np.radians(planner.solve_pose(np.r_[bottle,1.088], WRIST))
            actual_lift = np.array([np.radians(planner.solve_pose(np.r_[bottle,1.088+z], WRIST)) for z in np.linspace(0,.044,9)])
            raised = clearance_approach(planner,np.radians(approach[0]),grasp,.04)
            hits = []
            for u in np.linspace(0,1,81):
                contact = intersections(planner,meshes,interpolate(raised,u))
                if contact: hits.append(dict(fraction=float(u),meshes=contact))
            row.update(reference_ik_reachable=True, camera_actor_grasp_ik_reachable=True,
                sampled_approach_poses=81, bench_intersections=hits,
                ready=bool(margin>=.009 and np.linalg.norm(cup-bottle)>.07 and not hits))
        except ValueError as error:
            row.update(ready=False, error=str(error))
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(scope='Fixture support, exact reference IK through 165 degrees, actual actor-style 44 mm vertical lift and raised approach against fixed bench convex meshes. Excludes self/object collision, dynamics and learned trajectory extrapolation; not task success.',cases=rows)
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'proposed-plan.json').write_text(json.dumps(proposed,indent=2)+'\n')
    assert all(r['ready'] for r in rows), 'Resolve geometric setup failures before freezing'


if __name__=='__main__': main()
