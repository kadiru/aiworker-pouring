"""CPU IK and collision-envelope screen; no physics or policy success claim."""
import hashlib,json,os,sys
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/src','/workspace/learning/A-finish-20261106-v1/staged','/workspace']
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from preflight_finish_study import geometry,object_hits,intersections,vessel
from pxr import UsdGeom,UsdPhysics
from robot_pour import build_path,interpolate,ROOT as ROBOT
from pour_vessel_geometry import cup_profile
from wide_gripper_geometry import finger_meshes
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def main():
    plan=json.loads((ROOT/'plan.json').read_text());stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
    inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
    for prim in stage.Traverse():
        name=str(prim.GetPath())
        if '/right_arm/gripper_r_' not in name or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());m=inv@np.array(cache.GetLocalToWorldTransform(prim)).T
        local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local);meshes.append((6,name,local[hull.vertices],hull.equations))
    rows=[]
    for spec in [s for k in ('collect','perception_holdout','development','test') for s in plan[k]]:
        row=dict(name=spec['name'],passed=False)
        try:
            approach,_,_,own=build_path(stage,spec,include_planner=True);approach=np.radians(approach)
            cup=np.asarray(spec['cup_world']);profile=cup_profile(spec)
            points=np.array([[(r+.0044)*np.cos(a),(r+.0044)*np.sin(a),z+dz] for z,r in profile for dz in (-.002,.002) for a in np.arange(24)*2*np.pi/24])+cup
            shape=(points,ConvexHull(points).equations)
            if spec.get('calibration_only'):
                q=np.radians(own.solve(spec['calibration_hold_angle_deg'],1.)) if 'calibration_hold_angle_deg' in spec else approach[0]
                actual=meshes
                if 'calibration_hold_angle_deg' in spec:
                    assert 'calibration_finger_initial_deg' in spec
                    actual=meshes[:8]+finger_meshes(stage,np.radians(spec['calibration_finger_initial_deg']))
                row.update(bench=intersections(planner,actual,q),cup=object_hits(planner,actual,q,shape),scope='Authored stationary arm pose; articulated declared finger initialization')
                row['passed']=not row['bench'] and not row['cup']
            else:
                bottle=cup+[-spec['source_xy'][1],spec['source_xy'][0],0.]
                initial=object_hits(planner,meshes,approach[0],vessel(bottle,'bottle'));bench=[];hits=[]
                for u in np.linspace(0,1,41):
                    q=interpolate(approach,u)
                    if intersections(planner,meshes,q):bench.append(float(u))
                    if object_hits(planner,meshes,q,shape):hits.append(float(u))
                row.update(initial_bottle=initial,bench=bench,cup=hits,scope='Full commanded approach envelope')
                row['passed']=not initial and not bench and not hits
        except ValueError as e:row['error']=str(e)
        rows.append(row);print(json.dumps(row),flush=True)
        (ROOT/'geometry-preflight.json').write_text(json.dumps(dict(cases=rows,passed=all(r['passed'] for r in rows),complete=len(rows)==sum(len(plan[k]) for k in ('collect','perception_holdout','development','test')),
            plan_sha256=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),scope='IK and sampled conservative collision envelopes; does not establish dynamic grasp, fluid containment or task success.'),indent=2)+'\n')
    assert all(r['passed'] for r in rows)
if __name__=='__main__':main()
