"""Sampled CPU IK and robot/bottle/cup geometry, independent of fluid scoring."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
sys.path[:0]=[str(ROOT/'staged'),'/workspace/src','/workspace/learning/A-finish-20261106-v1/staged','/workspace']
from preflight_finish_study import geometry,object_hits,intersections
from robot_pour import build_path,FRAME,ROOT as ROBOT
from pour_vessel_geometry import cup_profile
from wide_gripper_geometry import finger_meshes
from pxr import UsdGeom,UsdPhysics
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('seed_pilot','seed_remaining','low_train','train','validation','holdout'),required=True);p.add_argument('--protocol',default='protocol.json');a=p.parse_args()
    assert Path(a.protocol).name==a.protocol and a.protocol.endswith('.json')
    protocol=json.loads((ROOT/a.protocol).read_text());cases=protocol.get(a.part)
    if a.part.startswith('seed_'):
        pilot=set(protocol['seed_pilot_names']);cases=[s for s in protocol['seed'] if (s['name'] in pilot)==(a.part=='seed_pilot')]
    assert cases;stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
    inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
    for prim in stage.Traverse():
        if '/right_arm/gripper_r_' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());m=inv@np.array(cache.GetLocalToWorldTransform(prim)).T;local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local)
        meshes.append((6,str(prim.GetPath()),local[hull.vertices],hull.equations))
    rows=[];path_cache={}
    for s in cases:
        key=json.dumps({k:s.get(k) for k in ('protocol','cup_world','source_xy','mouth_z','approach_clearance_m','home_clearance_m','calibration_motion')},sort_keys=True)
        if key not in path_cache:
            approach,_,_,own=build_path(stage,s,include_planner=True)
            if s.get('calibration_motion'):
                from pour_dynamic_motion import path
                times,joints=path(s,own)
            else:times,joints=np.array([0.]),np.asarray([approach[0]])
            path_cache[key]=(times,joints)
        times,joints=path_cache[key];profile=cup_profile(s);cup=np.asarray(s['cup_world'])
        points=np.array([[(r+.0044)*np.cos(t),(r+.0044)*np.sin(t),z+dz] for z,r in profile for dz in (-.002,.002) for t in np.arange(24)*2*np.pi/24])+cup
        shape=(points,ConvexHull(points).equations);actual=meshes
        if s.get('calibration_motion'):actual=meshes[:8]+finger_meshes(stage,np.radians(s['calibration_finger_initial_deg']))
        bench=[];hits=[];bottle_hits=[];table=[]
        for t,q in zip(times,np.radians(joints)):
            if intersections(planner,actual,q):bench.append(float(t))
            if object_hits(planner,actual,q,shape):hits.append(float(t))
            if s.get('calibration_motion'):
                from pour_dynamic_motion import motion_pose
                from bottle_pour import BOTTLE
                from scale_bottle_sat import convex_overlap
                base,rot,_=motion_pose(s,float(t));v=np.array([[(r+.001)*np.cos(phi),(r+.001)*np.sin(phi),z] for z,r in BOTTLE for phi in np.arange(16)*2*np.pi/16])
                world=(v@rot.T+base)@FRAME.T+cup
                if convex_overlap(world,points)['intersect']:bottle_hits.append(float(t))
                if world[:,2].min()<1.042:table.append(float(t))
        row=dict(name=s['name'],samples=len(times),bench=bench,cup=hits,bottle_cup=bottle_hits,bottle_table=table,passed=not(bench or hits or bottle_hits or table));rows.append(row)
        print(json.dumps(dict(name=s['name'],passed=row['passed'])),flush=True)
    report=dict(passed=all(r['passed'] for r in rows),complete=True,cases=rows,protocol_sha256=sha(ROOT/a.protocol),helper_sha256=sha(Path(__file__)),scope='Sampled authored calibration IK and conservative collision envelopes; no autonomous motion or fluid-success claim.')
    (ROOT/((a.protocol.removesuffix('.json')+'-' if a.protocol!='protocol.json' else '')+a.part+'-geometry.json')).write_text(json.dumps(report,indent=2)+'\n');assert report['passed']
if __name__=='__main__':main()
