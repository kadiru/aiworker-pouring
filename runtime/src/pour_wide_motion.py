"""Project a cloned joint target onto a calibrated pouring pose.

Inputs are RGB-localized initial object XY, known vessel/robot geometry and the
controller's own phase. No simulator object transform or particle state enters.
The learned target initializes IK; this geometry correction is programmed.
"""
import numpy as np
from bottle_pour import GRASP_Z, ease
from pour_wide_geometry import calibrated_pose
from robot_pour import FRAME, WRIST

UPRIGHT_SECONDS=1.1
DRAIN_HOLD_SECONDS=1.2
TRANSFER_SECONDS=2.0
RETURN_SECONDS=UPRIGHT_SECONDS+DRAIN_HOLD_SECONDS+TRANSFER_SECONDS
LATE_SHIFT_START_DEG=116.
LATE_SHIFT_END_DEG=122.


def late_mouth_offset(phase, amount):
    """Opt-in smooth lip shift toward cup center for the last emptying drops."""
    if not np.isfinite(amount) or not 0. <= amount <= .010:
        raise ValueError('Late mouth shift must be finite and between 0 and 10 mm')
    return amount*ease((phase-LATE_SHIFT_START_DEG)/(LATE_SHIFT_END_DEG-LATE_SHIFT_START_DEG))


def project_target(planner, prediction, source_xy, cup_xy, table_z, phase, mouth_z, centered=False, late_mouth_shift_m=0.):
    cup=np.r_[cup_xy,table_z]
    delta=np.asarray(source_xy)-np.asarray(cup_xy)
    initial=[delta[1],-delta[0]]
    base,rotation=calibrated_pose(phase,1.,initial,mouth_z,centered=centered)
    base[0]+=late_mouth_offset(phase,late_mouth_shift_m)
    tcp=cup+FRAME@(base+rotation@np.array([0.,0.,GRASP_Z]))
    orientation=FRAME@rotation@FRAME.T@WRIST
    planner.q=np.clip(np.asarray(prediction),planner.limits[0]+1e-7,planner.limits[1]-1e-7)
    q=np.radians(planner.solve_pose(tcp,orientation))
    if np.linalg.norm(planner.fk(q)[0]-tcp)>.001:
        raise ValueError('Projected pouring TCP misses calibrated target by over 1mm')
    return q


def centered_return_path(planner,current,phase,source_xy,cup_xy,table_z,lift,mouth_z,drain_hold_s=DRAIN_HOLD_SECONDS,late_mouth_shift_m=0.):
    """Timed upright-over-cup, drain pause, then raised translation to lift.

    Geometry and timing are engineered. Joint history and RGB-localized XY
    are the only episode-dependent inputs; no object/particle state is read.
    """
    from pour_workspace_path import clearance_approach
    from robot_pour import interpolate
    times=[0.];path=[np.asarray(current).copy()]
    for t in np.arange(.05,UPRIGHT_SECONDS+.0001,.05):
        angle=phase*(1-ease(t/UPRIGHT_SECONDS))
        path.append(project_target(planner,path[-1],source_xy,cup_xy,table_z,angle,mouth_z,centered=True,late_mouth_shift_m=late_mouth_shift_m));times.append(float(t))
    assert 0. <= drain_hold_s <= DRAIN_HOLD_SECONDS
    departure=UPRIGHT_SECONDS+drain_hold_s
    for t in np.arange(UPRIGHT_SECONDS+.05,departure+.0001,.05):
        path.append(path[-1].copy());times.append(float(t))
    transit=clearance_approach(planner,path[-1],lift,clearance_m=.04,level_transit=True)
    for elapsed in np.arange(.05,TRANSFER_SECONDS+.0001,.05):
        path.append(interpolate(transit,ease(elapsed/TRANSFER_SECONDS)));times.append(float(departure+elapsed))
    return np.asarray(times),np.asarray(path)


def timed_target(times,path,t):
    return np.array([np.interp(t,times,path[:,j]) for j in range(7)])


class CenteredTerminal:
    """Programmed over-cup return followed by the learned terminal profiles."""
    def __init__(self,history,start,profile,planner,phase,source_xy,cup_xy,table_z,lift,mouth_z,drain_hold_s=DRAIN_HOLD_SECONDS,late_mouth_shift_m=0.):
        from pour_stop_sequence import TerminalPrimitive
        self.times,self.path=centered_return_path(planner,history[-1][1],phase,source_xy,cup_xy,table_z,lift,mouth_z,drain_hold_s,late_mouth_shift_m)
        self.start=start;self.profile=dict(profile,return_s=float(self.times[-1]))
        self.inner=TerminalPrimitive(history,start,self.profile);self.stage=0

    def act(self,t):
        elapsed=max(0.,t-self.start)
        if elapsed<self.times[-1]:return np.r_[timed_target(self.times,self.path,elapsed),np.full(4,np.pi/4)],'Returning upright'
        result=self.inner.act(t);self.stage=self.inner.stage;return result
