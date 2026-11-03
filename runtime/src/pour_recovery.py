"""Privileged training-only expert takeover of a physically held bottle.

No bottle/particle transforms are written. Only ordinary arm/finger targets
are issued. In-hand registration and volume stopping are explicitly teacher
inputs; this controller is never used for autonomous policy evaluation.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from bottle_pour import GRASP_Z, pose, ease
from robot_pour import ArmPath, FRAME, ROOT, interpolate


class RecoveryExpert:
    def __init__(self, robot, t):
        from pxr import UsdPhysics
        row = robot.last_observation
        if not robot.verified or not row.get('both_fingers') or row['bottle_base_height_m'] < .025:
            raise ValueError('Takeover requires a verified, currently held lifted bottle')
        if robot.stop_time is not None:
            raise ValueError('Policy already stopped; cannot label this as a pouring recovery')
        self.robot = robot
        self.start = t
        joints = [UsdPhysics.RevoluteJoint.Get(robot.stage, ROOT+'/joints/arm_r_joint'+str(i)) for i in range(1,8)]
        lift = UsdPhysics.PrismaticJoint.Get(robot.stage, ROOT+'/joints/lift_joint')
        self.planner = ArmPath([j.GetLocalPos0Attr().Get() for j in joints],
            [j.GetAxisAttr().Get() for j in joints],
            [(j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()) for j in joints],
            lift.GetLocalPos0Attr().Get(),robot.cup_world,robot.start_xy,robot.spec.get('mouth_z',.052))
        q = np.radians([s.Get() for s in robot.states[:7]])
        self.planner.q = q.copy()
        tcp, wrist = self.planner.fk(q)
        actual_position = np.asarray(row['bottle_pose_xyz'])
        actual_rotation = Rotation.from_quat(row['bottle_quat_xyzw']).as_matrix()
        self.relative_position = wrist.T @ (actual_position-tcp)
        self.relative_rotation = wrist.T @ actual_rotation
        self.angle = float(np.clip(row['bottle_tilt_deg'],0.,82.))
        canonical = FRAME @ Rotation.from_euler('y',self.angle,degrees=True).as_matrix() @ FRAME.T
        residual = canonical.T @ actual_rotation
        self.yaw = Rotation.from_euler('z',math.atan2(residual[1,0],residual[0,0])).as_matrix()
        canonical_position, canonical_rotation = self.bottle_pose(self.angle,1.)
        slerp = Slerp([0.,1.],Rotation.from_matrix(np.stack([actual_rotation,canonical_rotation])))
        align=[]
        for x in np.linspace(0.,1.,17):
            align.append(self.solve(actual_position*(1-x)+canonical_position*x,slerp(x).as_matrix()))
        self.align=np.asarray(align)
        self.planner.q=q.copy()
        self.tilt=np.asarray([self.solve(*self.bottle_pose(a,1.)) for a in np.linspace(0.,165.,56)])
        self.planner.q=np.radians(self.tilt[0])
        self.down=np.asarray([self.solve(*self.bottle_pose(0.,x)) for x in np.linspace(0.,1.,9)])
        self.retreat=np.linspace(self.down[0],robot.approach[0],17)
        self.initial_command=robot.command.copy()
        self.last_t=t
        self.stop_angle=None
        # This flag enables the volume stop only after the privileged takeover.
        robot.expert_active=True
        robot.reason=None;robot.stop_time=None
        robot.confirmed=0;robot.last_signal=None;robot.source_history.clear()
        self.evidence=dict(takeover_sim_s=t,initial_bottle_angle_deg=row['bottle_tilt_deg'],
            initial_bottle_pose_xyz=row['bottle_pose_xyz'],
            measured_joint_deg=[float(s.Get()) for s in robot.states],
            in_hand_position_m=self.relative_position.tolist(),
            in_hand_rotation=self.relative_rotation.tolist(),
            actuation='Physical joint drives only; no bottle attachment or pose/particle writes',
            supervision='Expert labels begin at takeover; learned prefix is not an action target')

    def bottle_pose(self,angle,lift):
        base,rot=pose(angle,lift,self.robot.start_xy,self.robot.spec.get('mouth_z',.052))
        base[2]+=.018*lift*(1-ease(angle/60.))
        return self.robot.cup_world+FRAME@base,FRAME@rot@FRAME.T@self.yaw

    def solve(self,position,rotation):
        wrist=rotation@self.relative_rotation.T
        tcp=position-wrist@self.relative_position
        return self.planner.solve_pose(tcp,wrist)

    def drive(self,t):
        robot=self.robot;elapsed=t-self.start
        closure=1.
        if robot.stop_time is None:
            if elapsed<1.:
                arm=interpolate(self.align,ease(elapsed))
            else:
                row=robot.last_observation or {}
                limit=6. if row.get('cup_inventory_fraction',0.)>.65 else 12.
                rate=20. if self.angle<82. else (10. if row.get('source_outflow_ml_s',0.)<limit else 0.)
                self.angle=min(165.,self.angle+rate*max(0.,t-self.last_t))
                arm=interpolate(self.tilt,self.angle/165.)
            robot.phase='Pouring from gripper'
        else:
            if self.stop_angle is None:self.stop_angle=self.angle
            elapsed=t-robot.stop_time
            self.angle=self.stop_angle*(1-ease(elapsed/.85))
            arm=interpolate(self.tilt,self.angle/165.)
            robot.phase='Returning upright'
            if elapsed>.95:
                arm=interpolate(self.down,1-ease((elapsed-.95)/.65))
                robot.phase='Setting bottle down'
            if elapsed>1.8:
                closure=1-ease((elapsed-1.8)/.35)
                arm=interpolate(self.retreat,ease((elapsed-2.2)/.5))
                robot.phase='Releasing bottle'
        # Limit the splice and every subsequent move in the same units as Actor.
        desired=np.r_[arm,np.full(4,45.*closure)]
        delta=np.clip(desired-robot.command,-np.r_[[math.degrees(.12)]*7,[math.degrees(.85)]*4],
                                            np.r_[[math.degrees(.12)]*7,[math.degrees(.85)]*4])
        robot.command=robot.command+delta
        robot.command_angle=self.angle
        self.last_t=t
        for drive,target in zip(robot.drives,robot.command):drive.Set(float(target))

    def complete(self,t):
        return self.robot.stop_time is not None and t>=self.robot.stop_time+3.
