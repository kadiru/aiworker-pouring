"""Training-only cup initialization inside its actual tapered interior."""
import numpy as np
import hashlib
from pathlib import Path
from bottle_pour import CUP,SIDES

def cup_points(spec):
    assert spec.get('calibration_only') and spec['calibration_version'] in (1,2)
    if spec['calibration_version']==2:
        path=Path(spec['calibration_particle_seed'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['calibration_particle_seed_sha256']
        with np.load(path,allow_pickle=False) as seed:points=seed['cup_local_points'].copy()
        assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
        top=CUP[0][0]+spec['calibration_initial_height']*(CUP[-1][0]-CUP[0][0])
        return points[points[:,2]<=top].astype(np.float32)
    spacing=spec['spacing'];pitch_fraction=spec.get('calibration_pitch_fraction',.85)
    assert .80<=pitch_fraction<=.85
    pitch=pitch_fraction*spacing
    fraction=spec['calibration_initial_height']
    assert .7<=fraction<=1.
    top=CUP[0][0]+fraction*(CUP[-1][0]-CUP[0][0])
    points=[]
    for z in np.arange(CUP[0][0]+1.5*spacing,top-spacing/2,pitch):
        radius=np.interp(z,[p[0] for p in CUP],[p[1] for p in CUP])*np.cos(np.pi/SIDES)-1.5*spacing
        n=int(radius/pitch)
        points.extend((i*pitch,j*pitch,z) for i in range(-n,n+1) for j in range(-n,n+1) if (i*i+j*j)*pitch**2<radius**2)
    return np.asarray(points,dtype=np.float32)
