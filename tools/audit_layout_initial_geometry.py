"""Post-hoc setup audit: include the four finger meshes omitted by preflight.

This never changes or reruns the frozen test scenes. Native first-sample contacts
remain the evidence for whether a setup collision actually occurred.
"""
import json
from check_workspace_clearance import geometry
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics
from robot_pour import build_path,ROOT as ROBOT
from pour_layout_variety import ROOT,plan

stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
added=[]
for p in stage.Traverse():
    name=str(p.GetPath())
    if '/right_arm/gripper_r_' not in name or not p.IsA(UsdGeom.Mesh) or not p.HasAPI(UsdPhysics.CollisionAPI):continue
    points=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get());m=inv@np.array(cache.GetLocalToWorldTransform(p)).T
    local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local)
    meshes.append((6,name,local[hull.vertices],hull.equations));added.append(name)
assert len(added)==4
p=plan();home=np.radians(build_path(stage,p['test'][0])[0][0]);xyz=planner.shoulder.copy();rot=np.eye(3);trans=[]
for delta,axis,angle in zip(planner.offsets,planner.axes,home):
    xyz=xyz+rot@delta;rot=rot@Rotation.from_rotvec(axis*angle).as_matrix();trans.append((xyz.copy(),rot.copy()))
rows=[]
for layout in p['layouts_world_xy']:
    hits=[]
    for kind,profile in [('bottle',[(.004,.025),(.082,.025),(.107,.009),(.135,.009)]),('cup',[(.004,.020),(.038,.024)])]:
        center=np.r_[layout[kind],1.04]
        points=np.array([[r*np.cos(a),r*np.sin(a),z] for z,r in profile for a in np.arange(32)*2*np.pi/32])+center
        eqv=ConvexHull(points).equations
        for i,name,local,eq in meshes:
            xyz,rot=trans[i];mp=local@rot.T+xyz
            if not (np.minimum(mp.max(0),points.max(0))>np.maximum(mp.min(0),points.min(0))+1e-7).all():continue
            normal=eq[:,:3]@rot.T;rhs=-eq[:,3]+normal@xyz
            result=linprog(np.zeros(3),A_ub=np.vstack((normal,eqv[:,:3])),b_ub=np.r_[rhs,-eqv[:,3]],bounds=[(None,None)]*3,method='highs')
            assert result.status in (0,2),result.message
            if result.success:hits.append(dict(object=kind,mesh=name))
    row=dict(name=layout['name'],initial_envelope_intersections_including_fingers=hits,clear=not hits)
    rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'posthoc-complete-hand-geometry.json').write_text(json.dumps(dict(
    scope='Post-hoc CPU geometry, after tests began. Adds all four right finger collision meshes at their authored open pose relative to link7. Does not replace native evidence or rewrite the earlier preflight.',
    checked_mesh_count=len(meshes),added_finger_meshes=added,cases=rows),indent=2)+'\n')
