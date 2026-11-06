"""Raised Cartesian approach from joint/FK and fixed bench calibration only."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def clearance_approach(planner, home, grasp, clearance_m=.04, knots=41, level_transit=False):
    """Radians in/out. Keep both original endpoint joint configurations.

    A straight joint interpolation can dip the wrist into the worktop even
    when both endpoints are reachable. Raise, translate, then descend, using
    the interpolated endpoint posture to keep the redundant IK branch smooth.
    This is a candidate path; native collision/contact checks remain necessary.
    """
    home=np.asarray(home,dtype=float);grasp=np.asarray(grasp,dtype=float)
    assert 0 < clearance_m <= .10 and knots >= 13
    a,ra=planner.fk(home);b,rb=planner.fk(grasp)
    height=max(a[2],b[2])+clearance_m
    orientation=Slerp([0.,1.],Rotation.from_matrix(np.stack((ra,rb))))
    path=[home.copy()]
    for u in np.linspace(0.,1.,knots)[1:-1]:
        if u < .25:
            v=u/.25;rise=height-a[2] if level_transit else clearance_m
            xyz=a+[0,0,rise*(v*v*(3-2*v))]
        elif u < .75:
            v=(u-.25)/.5;v=v*v*(3-2*v)
            xyz=a*(1-v)+b*v+[0,0,clearance_m]
            if level_transit:xyz[2]=height
        else:
            v=(u-.75)/.25;v=v*v*(3-2*v)
            rise=height-b[2] if level_transit else clearance_m
            xyz=b+[0,0,rise*(1-v)]
        planner.q=home*(1-u)+grasp*u
        path.append(np.radians(planner.solve_pose(xyz,orientation(u).as_matrix())))
    path.append(grasp.copy());planner.q=grasp.copy()
    return np.asarray(path)
