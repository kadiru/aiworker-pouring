"""Screen approach, held bottle and side return with declared vessel geometry."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--study", choices=("sized-autonomy", "sized-autonomy-r1", "bottle-autonomy",
                                       "bottle-empty-autonomy", "mixed-vessel-autonomy", "tall-empty-r1"),
                    default="sized-autonomy")
STUDY = ROOT/parser.parse_args().study
sys.path[:0] = [str(STUDY/"src"), "/workspace/learning/A-finish-20261106-v1/staged", "/workspace"]
from preflight_finish_study import geometry, intersections, object_hits, vessel
from bottle_pour import GRASP_Z, WALL, initial_points
from pour_bottle_geometry import declared_geometry, profile_for
from pour_guard_trajectory import target
from pour_side_return import paths
from pour_size_autonomy import sha
from pour_vessel_geometry import cup_profile
from pour_wide_motion import project_target
from robot_pour import FINGERS, FRAME, WRIST, build_path
from wide_gripper_geometry import finger_meshes


def shape(center, profile, margin):
    angle = np.linspace(0, 2*np.pi, 24, endpoint=False)
    points = np.array([[(r+margin)*np.cos(a),(r+margin)*np.sin(a),z+dz]
                       for z,r in profile for dz in (-.002,.002) for a in angle])+center
    return points, ConvexHull(points).equations


def bottle_shell(arm, joint, local):
    tcp, wrist = arm.fk(joint)
    rotation = wrist @ WRIST.T
    return (local - [0.,0.,GRASP_Z]) @ rotation.T + tcp


def scan(stage, spec, saved, opened, closed):
    approach,lift,_,arm=build_path(stage,spec,include_planner=True)
    approach,lift=np.radians(approach),np.radians(lift)
    home=approach[0]
    cup=np.array(spec["cup_world"])
    source=cup+FRAME@[*spec["source_xy"],0.]
    cup_shape=shape(cup,cup_profile(spec),.0044)
    bottle_profile=profile_for(spec)
    bottle=shape(np.zeros(3),bottle_profile,WALL)[0]
    resized=spec.get("bottle_size_autonomy")
    if resized:
        assert saved["bottle_geometry"] == spec["declared_bottle_geometry"] == declared_geometry(spec)
        initial_points(spec)
        source_shape=shape(source+[0.,0.,.0002],bottle_profile,WALL)
    else:
        source_shape=vessel(source,"bottle")
    failures=[]
    if object_hits(arm,opened,home,source_shape):
        failures.append("ready hand intersects source")
    first_contact=None
    if resized:
        for degrees in (32.5,35.,37.5,40.,42.5):
            fingers=finger_meshes(stage,np.full(4,np.radians(degrees)))
            hits=object_hits(arm,fingers,approach[-1],source_shape)
            if all(any(tip in hit for hit in hits) for tip in FINGERS):
                first_contact=degrees
                break
        if first_contact is None:
            failures.append("no opposed-tip grasp before full closure")
    def check(label, joint, meshes, carrying=False, placed=None):
        if intersections(arm,meshes,joint):
            failures.append(label+": bench")
        if object_hits(arm,meshes,joint,cup_shape):
            failures.append(label+": arm/cup")
        if carrying:
            shell=bottle_shell(arm,joint,bottle)
            points, cup_planes = cup_shape
            if np.all(np.minimum(shell.max(0),points.max(0)) >= np.maximum(shell.min(0),points.min(0))):
                planes = np.vstack((ConvexHull(shell).equations, cup_planes))
                overlap = linprog(np.zeros(3), A_ub=planes[:, :3], b_ub=-planes[:, 3],
                                  bounds=[(None,None)]*3, method="highs")
                assert overlap.status in (0,2), overlap.message
                if overlap.success:
                    failures.append(label+": bottle/cup")
            if shell[:,2].min()<cup[2]-.002:
                failures.append(label+": bottle/bench")
        if placed is not None:
            placed_shape=shape(placed,bottle_profile,WALL) if resized else vessel(placed,"bottle",margin=0.)
            if object_hits(arm,meshes,joint,placed_shape):
                failures.append(label+": arm/placed bottle")

    for joint in approach:
        check("approach",joint,opened)
        if resized and object_hits(arm,opened,joint,source_shape):
            failures.append("approach: open hand/source")
    for joint in lift:
        check("lift",joint,closed,True)
    profile=saved["pour_primitive"]
    phases=sorted(set(np.arange(0.,132.,3.).tolist()+[60.,80.,90.,100.,110.,131.]))
    projected={}
    for phase in phases:
        projected[phase]=project_target(arm,target(profile,lift[-1],cup[:2],phase),
                                       source[:2],cup[:2],cup[2],phase,saved["pour_mouth_z"],
                                       late_mouth_shift_m=saved["late_mouth_shift_m"],
                                       lip_height=bottle_profile[-1][0])
        check("pour "+str(phase),projected[phase],closed,True)
    returns=[]
    for phase in (60.,80.,90.,100.,110.,131.):
        _,upright,place,retreat,info=paths(arm,projected[phase],home,cup[:2],cup[2],saved["side_return"],
                                         bottle_profile=bottle_profile)
        for joint in upright:
            check("upright "+str(phase),joint,closed,True)
        for joint in place:
            check("placement "+str(phase),joint,closed,True)
        placed=np.r_[info["placement_xy"],cup[2]]
        for index,joint in enumerate(retreat):
            check("retreat "+str(phase),joint,opened,placed=placed if index else None)
        returns.append(dict(phase_deg=phase,side=info["side"],step_rad=info["max_planned_joint_step_rad"]))
    row=dict(name=spec["name"],passed=not failures,failures=sorted(set(failures)),
             sampled_outbound_poses=len(projected),returns=returns)
    if resized:
        row.update(first_opposed_tip_contact_deg=first_contact,
                   bottle_geometry=saved["bottle_geometry"])
    return row


def main():
    torch.set_num_threads(1)
    plan_path=STUDY/"plan.json"
    plan=json.loads(plan_path.read_text())
    stage,_,base=geometry()
    opened=base+finger_meshes(stage,[0.,0.,0.,0.])
    variants=[finger_meshes(stage,v) for v in ((.64,.75,.64,.75),(.75,.95,.75,.95),(.64,.95,.64,.95),(.75,.75,.75,.75))]
    closed=list(base)
    for i in range(4):
        vertices=np.concatenate([v[i][2] for v in variants])
        hull=ConvexHull(vertices)
        closed.append((6,variants[0][i][1],vertices[hull.vertices],hull.equations))
    cases=[]
    for spec in plan["development"]+plan["test"]:
        try:
            assert sha(spec["checkpoint"])==plan["checkpoints"][spec["checkpoint"]]
            saved=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
            row=scan(stage,spec,saved,opened,closed)
        except ValueError as error:
            row=dict(name=spec["name"],passed=False,failures=[str(error)])
        cases.append(row)
        print(json.dumps(row),flush=True)
    result=dict(passed=all(r["passed"] for r in cases),cases=cases,plan_sha256=sha(plan_path),
                sources={p.name:sha(p) for p in sorted((STUDY/"src").glob("*.py"))},
                scope="Sampled robot, gripper and held-bottle clearance with actual cup geometry")
    with (STUDY/"geometry.json").open("x") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    assert not torch.cuda.is_initialized()


if __name__=="__main__":
    main()
