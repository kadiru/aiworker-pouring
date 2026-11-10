"""CPU-only bottle-size grasp envelope screen; no runtime bottle scaling."""
import os
import sys
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/src','/workspace/learning/A-finish-20261106-v1/staged','/workspace']
from preflight_finish_study import geometry, object_hits, intersections, transforms
from scale_bottle_sat import convex_overlap
from wide_gripper_geometry import finger_meshes, REST_ERRORS
from robot_pour import FRAME, WRIST, FINGERS
from scipy.spatial import ConvexHull
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/learning/A-bottle-size-20261110-v1')


def main():
    inventory=json.loads((ROOT/'session-plan.json').read_text())
    spec=json.loads(Path('/workspace/learning/A-scale-20261109-v1/plan.json').read_text())['pilot'][1]
    stage,planner,arm=geometry()
    center=np.asarray(spec['cup_world'])+FRAME@[*spec['source_xy'],.0002]
    angles=np.arange(0.,45.001,2.5)
    fingers={float(a):finger_meshes(stage,np.full(4,np.radians(a))) for a in angles}
    rows=[]
    assert not (ROOT/'bottle-grasp-envelope-sat.json').exists()
    for s in inventory['development_shapes']:
        profile=s['profile'];grasp=s['grasp_height_m']
        target=center+[0.,0.,grasp];q=np.radians(planner.solve_pose(target,WRIST))
        actual,rotation=planner.fk(q)
        points=np.array([[(r+.0024)*np.cos(a),(r+.0024)*np.sin(a),z]
                         for z,r in profile for a in np.arange(40)*2*np.pi/40])+center
        shape=(points,ConvexHull(points).equations)
        contact=[]
        for a in angles:
            try:
                hits=object_hits(planner,fingers[float(a)],q,shape)
                contact.append(dict(command_deg=float(a),finger_hulls=hits,
                                    opposed_tips_intersect=all(any(tip in hit for hit in hits) for tip in FINGERS)))
            except AssertionError as error:
                # Preserve the LP failure and corroborate with an independent
                # face/edge separating-axis test; an error is never clearance.
                poses=transforms(planner,q);checks=[];hits=[]
                for link,name,local,_ in fingers[float(a)]:
                    xyz,rot=poses[link];result=convex_overlap(local@rot.T+xyz,points)
                    checks.append(dict(mesh=name,**result))
                    if result['intersect']:hits.append(name)
                contact.append(dict(command_deg=float(a),solver_error=str(error),independent_sat=checks,
                                    finger_hulls=hits,opposed_tips_intersect=all(any(tip in hit for hit in hits) for tip in FINGERS)))
        open_hits=object_hits(planner,arm+fingers[0.],q,shape)
        palm_hits=object_hits(planner,arm,q,shape)
        bench=intersections(planner,arm+fingers[0.],q)
        bracket=next((r['command_deg'] for r in contact if r['opposed_tips_intersect']),None)
        row=dict(height_scale=s['height_scale'],body_diameter_scale=s['body_scale'],
                 outer_body_diameter_mm=s['outer_body_diameter_m']*1000,lip_height_mm=s['lip_height_m']*1000,
                 grasp_height_mm=s['grasp_height_m']*1000,grasp_position_error_mm=float(np.linalg.norm(actual-target)*1000),
                 open_bottle_intersections=open_hits,palm_bottle_intersections=palm_hits,open_bench_intersections=bench,
                 first_sampled_opposed_tip_intersection_deg=bracket,finger_closure_sweep=contact,
                 necessary_geometry_pass=not open_hits and not palm_hits and not bench and bracket is not None
                    and not any('solver_error' in r and 'independent_sat' not in r for r in contact))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='finger_closure_sweep'}),flush=True)
    report=dict(cases=rows,inventory_sha256=hashlib.sha256((ROOT/'session-plan.json').read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),joint_rest_frame_errors=REST_ERRORS,
                scope='One declared table layout, static upright grasp. Exact authored finger-joint kinematics and convex bottle envelope at 2.5 degree closure increments. Geometric intersections bracket potential contact; they do not establish stable grasp, force, friction, mass/inertia, dynamic clearance, tracking or autonomous pouring. Runtime bottle scaling remains disabled.')
    report.update(independent_sat_helper_sha256=hashlib.sha256(Path(__file__).with_name('scale_bottle_sat.py').read_bytes()).hexdigest())
    with (ROOT/'bottle-grasp-envelope-sat.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':main()
