"""Articulate finger hulls from USD joint frames for closed-grasp checks."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics
from robot_pour import ROOT
REST_ERRORS={}


def local_frame(position,rotation):
    im=rotation.GetImaginary();q=[*im,rotation.GetReal()]
    m=np.eye(4);m[:3,:3]=Rotation.from_quat(q).as_matrix();m[:3,3]=position;return m


def finger_meshes(stage,angles):
    cache=UsdGeom.XformCache();wrist=stage.GetPrimAtPath(ROOT+'/right_arm/arm_r_link7')
    inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(wrist)).T)
    actual={str(wrist.GetPath()):np.eye(4)};zero=dict(actual);bodies=[]
    for i,angle in enumerate(angles,1):
        j=UsdPhysics.RevoluteJoint.Get(stage,ROOT+'/joints/gripper_r_joint'+str(i))
        body0=str(j.GetBody0Rel().GetTargets()[0]);body1=str(j.GetBody1Rel().GetTargets()[0]);bodies.append(body1)
        if body0 not in actual:
            actual[body0]=inv@np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0))).T;zero[body0]=actual[body0]
        a=local_frame(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get())
        b=local_frame(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())
        motion=np.eye(4);axis=np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]
        motion[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()
        actual[body1]=actual[body0]@a@motion@np.linalg.inv(b);zero[body1]=zero[body0]@a@np.linalg.inv(b)
        rest=inv@np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1))).T
        position_error=float(np.linalg.norm(rest[:3,3]-zero[body1][:3,3]))
        rotation_error=float(np.degrees(Rotation.from_matrix(rest[:3,:3]@zero[body1][:3,:3].T).magnitude()))
        REST_ERRORS[body1]=dict(position_m=position_error,rotation_deg=rotation_error)
        # CAD-authored rest transforms have small discrepancies from the joint
        # constraints. PhysX enforces the latter; report, bound and preserve
        # the discrepancy rather than treating authored rest as exact FK.
        assert position_error<.002 and rotation_error<.2,(body1,REST_ERRORS[body1])
    meshes=[]
    for name in bodies:
        body=stage.GetPrimAtPath(name);to_body=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)
        for prim in body.GetChildren():
            # Colliders are nested two levels below the rigid finger body.
            from pxr import Usd
            for mesh in Usd.PrimRange(prim):
                if not mesh.IsA(UsdGeom.Mesh) or not mesh.HasAPI(UsdPhysics.CollisionAPI):continue
                transform=actual[name]@to_body@np.array(cache.GetLocalToWorldTransform(mesh)).T
                points=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
                local=points@transform[:3,:3].T+transform[:3,3];hull=ConvexHull(local)
                meshes.append((6,str(mesh.GetPath()),local[hull.vertices],hull.equations))
    assert len(meshes)==4
    return meshes
