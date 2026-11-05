"""Legacy eight-mesh arm check; omitted fingers, so NOT full-hand clearance.

Preserved for the original preflight's provenance. New layouts must use the
complete-hand checks in preflight_layout_extra.py instead.
"""
import json
from check_workspace_clearance import geometry
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from robot_pour import build_path
from pour_layout_variety import ROOT

stage, planner, meshes = geometry()
assert not (ROOT/'selection.json').exists(), 'Do not overwrite frozen legacy preflight evidence'
plan = json.loads((ROOT/'proposed-plan.json').read_text())
home = np.radians(build_path(stage,plan['test'][0])[0][0])
xyz=planner.shoulder.copy();rot=np.eye(3);trans=[]
for delta,axis,angle in zip(planner.offsets,planner.axes,home):
    xyz=xyz+rot@delta;rot=rot@Rotation.from_rotvec(axis*angle).as_matrix()
    trans.append((xyz.copy(),rot.copy()))
rows=[]
for layout in plan['layouts_world_xy']:
    hits=[]
    for kind,profile in [('bottle',[(.004,.025),(.082,.025),(.107,.009),(.135,.009)]),
                         ('cup',[(.004,.020),(.038,.024)])]:
        center=np.r_[layout[kind],1.04]
        points=np.array([[r*np.cos(a),r*np.sin(a),z] for z,r in profile for a in np.arange(32)*2*np.pi/32])+center
        eqv=ConvexHull(points).equations
        for i,name,local,eq in meshes:
            xyz,rot=trans[i];mesh_points=local@rot.T+xyz
            if not (np.minimum(mesh_points.max(0),points.max(0))>np.maximum(mesh_points.min(0),points.min(0))+1e-7).all():continue
            normal=eq[:,:3]@rot.T;rhs=-eq[:,3]+normal@xyz
            result=linprog(np.zeros(3),A_ub=np.vstack((normal,eqv[:,:3])),b_ub=np.r_[rhs,-eqv[:,3]],bounds=[(None,None)]*3,method='highs')
            assert result.status in (0,2), (layout['name'],kind,name,result.message)
            if result.success:hits.append(dict(object=kind,mesh=name))
    row=dict(name=layout['name'],initial_convex_envelope_intersections=hits,clear=not hits)
    rows.append(row);print(json.dumps(row),flush=True)
report=dict(scope='Initial open-hand robot collision meshes versus conservative closed convex vessel envelopes, CPU only. No motion or native contact proof.',cases=rows)
(ROOT/'initial-object-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(r['clear'] for r in rows), 'Reposition intersecting fixtures before freezing'
