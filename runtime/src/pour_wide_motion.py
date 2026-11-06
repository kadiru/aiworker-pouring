"""Project a cloned joint target onto a calibrated pouring pose.

Inputs are RGB-localized initial object XY, known vessel/robot geometry and the
controller's own phase. No simulator object transform or particle state enters.
The learned target initializes IK; this geometry correction is programmed.
"""
import numpy as np
from bottle_pour import pose, GRASP_Z, ease
from robot_pour import FRAME, WRIST


def project_target(planner, prediction, source_xy, cup_xy, table_z, phase, mouth_z):
    cup=np.r_[cup_xy,table_z]
    delta=np.asarray(source_xy)-np.asarray(cup_xy)
    initial=[delta[1],-delta[0]]
    base,rotation=pose(phase,1.,initial,mouth_z)
    base[2]+=.018*(1-ease(phase/60.))
    tcp=cup+FRAME@(base+rotation@np.array([0.,0.,GRASP_Z]))
    orientation=FRAME@rotation@FRAME.T@WRIST
    planner.q=np.clip(np.asarray(prediction),planner.limits[0]+1e-7,planner.limits[1]-1e-7)
    q=np.radians(planner.solve_pose(tcp,orientation))
    if np.linalg.norm(planner.fk(q)[0]-tcp)>.001:
        raise ValueError('Projected pouring TCP misses calibrated target by over 1mm')
    return q
