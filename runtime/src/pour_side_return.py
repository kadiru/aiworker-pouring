"""Continuous sideways uprighting and side placement from RGB and own joints.

Known table/vessel dimensions and bounded IK are programmed constraints. No
simulator object transform, contact, fluid inventory or test label is read.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from bottle_pour import BOTTLE, GRASP_Z, WALL, ease
from robot_pour import WRIST
from pour_wide_motion import timed_target

TABLE_LO=np.array([.305,-.43]);TABLE_HI=np.array([.685,.03])
PLACEMENT_MARGIN=.035
OFFSETS=(.18,.165,.1524,.20,.23)


def placement_candidates(cup_xy):
    """Robot's right is -Y. Respect the full bottle footprint at both edges."""
    cup=np.asarray(cup_xy,dtype=float)
    for side,sign in (('right',-1.),('left',1.)):
        for offset in OFFSETS:
            for dx in (0.,-.025,.025,-.05):
                xy=cup+[dx,sign*offset]
                if np.all(xy>=TABLE_LO+PLACEMENT_MARGIN) and np.all(xy<=TABLE_HI-PLACEMENT_MARGIN):
                    yield dict(side=side,xy=xy,offset_m=float(np.linalg.norm(xy-cup)))


def solve(planner,previous,xyz,rotation):
    planner.q=np.clip(previous,planner.limits[0]+1e-7,planner.limits[1]-1e-7)
    q=np.radians(planner.solve_pose(xyz,rotation))
    actual,rot=planner.fk(q)
    if np.linalg.norm(actual-xyz)>.001 or Rotation.from_matrix(rotation@rot.T).magnitude()>.008:
        raise ValueError('Side return IK misses position/orientation tolerance')
    return q


def paths(planner,current,home,cup_xy,table_z,config,*,bottle_profile=BOTTLE):
    """The bottle base translates throughout the orientation interpolation.

    A raised arc gives the carried bottle clearance even for a left fallback.
    No intermediate upright pose over the cup or separate drain dwell exists.
    """
    duration=float(config.get('upright_seconds',2.4))
    place_s=float(config.get('place_seconds',1.8));retreat_s=float(config.get('retreat_seconds',2.6))
    assert 1.8<=duration<=4 and 1.5<=place_s<=3 and 2<=retreat_s<=4
    rim_z=float(config.get('rim_z_m', .038))
    early_lift=config.get('early_lift_arc',False)
    current=np.asarray(current);tcp,r0=planner.fk(current)
    # At upright the TCP orientation is WRIST; its relative rotation is the
    # physical vessel rotation. This is the same fixed calibration as pouring.
    b0=tcp-(r0@WRIST.T)@np.array([0.,0.,GRASP_Z])
    orientation=Slerp([0.,1.],Rotation.from_matrix(np.stack((r0,WRIST))))
    a=np.arange(40)*2*np.pi/40
    shell=np.array([[(r+WALL)*np.cos(t),(r+WALL)*np.sin(t),z] for z,r in bottle_profile for t in a])
    attempts=[]
    for choice in placement_candidates(cup_xy):
        try:
            end=np.r_[choice['xy'],table_z+.075]
            result=[current.copy()];times=[0.];min_margin=float('inf')
            for t in np.arange(.05,duration+.0001,.05):
                u=ease(t/duration);r=orientation(u).as_matrix();br=r@WRIST.T
                base=(1-u)*b0+u*end
                arc_progress=t/duration if early_lift else u
                base[2]+=.025*np.sin(np.pi*arc_progress)**2
                # Conservative swept vessel lower bound over receiving rim.
                lower=float((shell@br.T)[:,2].min())
                base[2]=max(base[2],table_z+rim_z+.012-lower)
                min_margin=min(min_margin,base[2]+lower-table_z-rim_z)
                result.append(solve(planner,result[-1],base+br@[0.,0.,GRASP_Z],r));times.append(float(t))
            place=[result[-1].copy()]
            start_tcp=planner.fk(place[0])[0];end_tcp=np.r_[choice['xy'],table_z+GRASP_Z+.0005]
            for u in np.linspace(0,1,37)[1:]:
                place.append(solve(planner,place[-1],start_tcp*(1-u)+end_tcp*u,WRIST))
            from pour_workspace_path import clearance_approach
            retreat=clearance_approach(planner,place[-1],home,clearance_m=.09,knots=61,level_transit=True)
            result=np.asarray(result);place=np.asarray(place)
            # Bound planned 20Hz joint commands before accepting a candidate.
            from pour_stop_sequence import path_at
            peak=float(np.abs(np.diff(result,axis=0)).max())
            if peak>.095 and config.get('retime_upright'):
                # Stretch the same piecewise linear path to bound every 20 Hz
                # step, including the initial rise from the measured pose.
                stretch=peak/.094
                if duration*stretch>4.:
                    raise ValueError('Measured upright path would require more than 4 seconds')
                times=np.asarray(times)*stretch
                peak/=stretch
            for path,seconds in ((place,place_s),(retreat,retreat_s)):
                q=np.array([path_at(path,ease(t/seconds)) for t in np.arange(0,seconds+.0001,.05)])
                peak=max(peak,float(np.abs(np.diff(q,axis=0)).max()))
            if peak>.095:raise ValueError(f'Planned step {peak:.5f} exceeds .095 rad')
            info=dict(side=choice['side'],placement_xy=choice['xy'].tolist(),cup_xy=list(cup_xy),
                offset_m=choice['offset_m'],simultaneous_uprighting=True,upright_seconds=float(times[-1]),
                early_lift_arc=early_lift,
                max_planned_joint_step_rad=peak,minimum_carried_bottom_above_rim_m=min_margin,
                rejected_candidates=attempts)
            return np.asarray(times),result,place,retreat,info
        except ValueError as error:
            attempts.append(dict(side=choice['side'],xy=choice['xy'].tolist(),reason=str(error)))
    raise ValueError('No reachable side placement: '+str(attempts))


class SideTerminal:
    def __init__(self,history,start,profile,planner,cup_xy,table_z,config,*,bottle_profile=BOTTLE):
        from pour_stop_sequence import TerminalPrimitive
        self.times,self.path,place,retreat,self.diagnostic=paths(
            planner,history[-1][1],history[0][1],cup_xy,table_z,config,bottle_profile=bottle_profile)
        self.start=start;self.stage=0
        self.profile=dict(profile,return_s=float(self.times[-1]),place_s=config.get('place_seconds',1.8),
            release_s=max(.6,profile['release_s']),retreat_s=config.get('retreat_seconds',2.6))
        self.inner=TerminalPrimitive(history,start,self.profile)
        self.inner.place_path=place;self.inner.retreat_path=retreat

    def act(self,t):
        elapsed=max(0.,t-self.start)
        if elapsed<self.times[-1]:
            return np.r_[timed_target(self.times,self.path,elapsed),np.full(4,np.pi/4)],'Returning upright'
        result=self.inner.act(t);self.stage=self.inner.stage;return result
