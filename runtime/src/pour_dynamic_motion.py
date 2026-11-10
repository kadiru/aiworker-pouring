"""Bounded authored bottle/hand motion for perception calibration, not a grasp."""
import numpy as np
from scipy.spatial.transform import Rotation
from pour_wide_geometry import calibrated_pose

def motion_pose(spec,t):
    if not spec.get('calibration_only') or spec.get('policy_evaluation') or spec.get('calibration_fill'):
        raise ValueError('Motion fixture is calibration-only and cannot emit liquid')
    c=spec['calibration_motion'];maximum=float(c['maximum_angle_deg'])
    shift=float(c['lateral_amplitude_m']);start=float(c['start_s']);duration=float(c['duration_s'])
    if not np.isfinite([maximum,shift,start,duration,t]).all() or not (45<=maximum<=85 and .01<=shift<=.025 and .3<=start<=.8 and 2<=duration<=4):
        raise ValueError('Motion fixture outside declared angle, speed or travel bounds')
    if spec['calibration_hold_angle_deg']!=0:raise ValueError('Begin upright beside the cup for initial image localization')
    u=float(np.clip((t-start)/duration,0,1));smooth=u*u*(3-2*u);wave=np.sin(2*np.pi*smooth)
    angle=maximum*.5*(1-np.cos(2*np.pi*smooth))
    position,rotation=calibrated_pose(angle,1.,spec['source_xy'],spec['mouth_z'])
    # Move across the tilt plane so translation cannot cancel the bottle's arc.
    position=position.copy();position[1]+=shift*wave
    return position,rotation,float(angle)

def path(spec,planner):
    from robot_pour import FRAME,WRIST,GRASP_Z
    c=spec['calibration_motion'];times=np.linspace(0,spec['max_seconds'],81);joints=[]
    for t in times:
        base,rot,_=motion_pose(spec,t)
        target=np.asarray(spec['cup_world'])+FRAME@(base+rot@[0.,0.,GRASP_Z])
        joints.append(planner.solve_pose(target,FRAME@rot@FRAME.T@WRIST))
    return times,np.asarray(joints)

def apply(robot,t):
    from pxr import Gf
    from robot_pour import FRAME
    base,rot,angle=motion_pose(robot.spec,t);rotation=FRAME@rot@FRAME.T
    robot.position.Set(Gf.Vec3d(*(robot.cup_world+FRAME@base)))
    q=Rotation.from_matrix(rotation).as_quat();robot.orientation.Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
    times,joints=robot.calibration_motion_path
    command=np.array([np.interp(t,times,joints[:,i]) for i in range(7)])
    robot.command=np.r_[command,np.full(4,45.)];robot.command_angle=angle
    robot.phase='Authored moving-bottle perception calibration'
    for drive,target in zip(robot.drives,robot.command):drive.Set(float(target))
