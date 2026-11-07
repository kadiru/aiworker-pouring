"""Training-only cup initialization inside its actual tapered interior."""
import numpy as np
from bottle_pour import CUP,SIDES

def cup_points(spec):
    assert spec.get('calibration_only') and spec['calibration_version']==1
    spacing=spec['spacing'];pitch=.85*spacing
    fraction=spec['calibration_initial_height']
    assert .7<=fraction<=1.
    top=CUP[0][0]+fraction*(CUP[-1][0]-CUP[0][0])
    points=[]
    for z in np.arange(CUP[0][0]+1.5*spacing,top-spacing/2,pitch):
        radius=np.interp(z,[p[0] for p in CUP],[p[1] for p in CUP])*np.cos(np.pi/SIDES)-1.5*spacing
        n=int(radius/pitch)
        points.extend((i*pitch,j*pitch,z) for i in range(-n,n+1) for j in range(-n,n+1) if (i*i+j*j)*pitch**2<radius**2)
    return np.asarray(points,dtype=np.float32)
