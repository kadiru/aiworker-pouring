"""Calibrated cup-rim geometry in RGB; no runtime object poses or particle state."""
import numpy as np
from scipy.ndimage import label,map_coordinates
from pour_guard_vision import plane_point,project

def project_many(points,cal):
    t=np.asarray(cal['T']);k=np.asarray(cal['K']);v=(points-t[:3,3])@t[:3,:3]
    return np.stack((k[0,0]*v[...,0]/-v[...,2]+k[0,2],k[1,2]-k[1,1]*v[...,1]/-v[...,2]),-1)

def locate_rim(image,bank):
    cal=bank['calibration'];g=bank.get('geometry',dict(rim_z_m=.038,rim_radius_m=.024,bottom_z_m=.004))
    rgb=np.asarray(image,dtype=np.float32)/255.;r,green,b=rgb.transpose(2,0,1)
    red=(r>1.4*green+.025)&(r>1.2*b+.02)&(r>.12)
    lo=np.array(bank['xy_min']);hi=np.array(bank['xy_max'])
    components,count=label(red);areas=np.bincount(components.ravel());areas[0]=0
    centers=[]
    for key in np.argsort(areas)[-8:][::-1]:
        if areas[key]<20:continue
        yy,xx=np.nonzero(components==key);uv=np.array([np.median(xx),np.median(yy)])
        guess=plane_point(uv,cal['table_z']+(g['bottom_z_m']+g['rim_z_m'])/2,cal)[:2]
        if np.any(guess<lo-.03) or np.any(guess>hi+.03):continue
        center=np.r_[guess,cal['table_z']+g['rim_z_m']]
        diameter=abs(project(center+np.asarray(cal['T'])[:3,0]*(g['rim_radius_m']+.0024),cal)[0]-project(center-np.asarray(cal['T'])[:3,0]*(g['rim_radius_m']+.0024),cal)[0])
        if xx.max()-xx.min()<diameter*bank.get('min_red_width_fraction',.5):continue
        centers.append((int(areas[key]),guess))
    if not centers:return project(np.r_[(lo+hi)/2,cal['table_z']+g['rim_z_m']],cal),0.
    guess=max(centers,key=lambda c:c[0])[1]
    gray=rgb@np.array([.299,.587,.114]);gy,gx=np.gradient(gray)
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    offsets=np.array(bank.get('radial_offsets_m',[.0018,.0024,.003]))
    depth=bank.get('rim_feature_depth_m',0.)
    taper=(g['rim_radius_m']-g.get('profile',[[0.,g['rim_radius_m']]])[0][1])/(g['rim_z_m']-g['bottom_z_m'])
    radii=g['rim_radius_m']-depth*taper+offsets
    ring=np.stack((radii[:,None]*np.cos(angles),radii[:,None]*np.sin(angles),np.zeros((len(radii),len(angles)))),-1)
    def scores(xy):
        center=np.c_[xy,np.full(len(xy),cal['table_z']+g['rim_z_m']-depth)]
        uv=project_many(center[:,None,None,:]+ring[None],cal)
        tangent=np.roll(uv,-1,axis=2)-np.roll(uv,1,axis=2)
        normal=np.stack((tangent[...,1],-tangent[...,0]),-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
        coords=[uv[...,1].ravel(),uv[...,0].ravel()]
        xx=map_coordinates(gx,coords,order=1,mode='constant').reshape(uv.shape[:-1])
        yy=map_coordinates(gy,coords,order=1,mode='constant').reshape(uv.shape[:-1])
        edge=np.abs(xx*normal[...,0]+yy*normal[...,1]).max(1)
        edge=np.minimum(edge,bank.get('gradient_clip',.15))
        # Average across the contour; a single strong bottle/gripper edge is
        # insufficient. Keep only the stronger 75% for partial hand occlusion.
        return np.sort(edge,axis=1)[:,24:].mean(1)
    low=np.maximum(lo,guess-.03);high=np.minimum(hi,guess+.03)
    if np.any(high<low):return project(np.r_[guess,cal['table_z']+g['rim_z_m']],cal),0.
    xx,yy=np.meshgrid(np.arange(low[0],high[0]+.0005,.001),np.arange(low[1],high[1]+.0005,.001))
    grid=np.c_[xx.ravel(),yy.ravel()];values=scores(grid);best=grid[int(values.argmax())]
    xx,yy=np.meshgrid(np.arange(-.001,.00101,.0002),np.arange(-.001,.00101,.0002));fine=best+np.c_[xx.ravel(),yy.ravel()]
    fine=fine[(fine>=lo).all(1)&(fine<=hi).all(1)];values=scores(fine);i=int(values.argmax())
    return project(np.r_[fine[i],cal['table_z']+g['rim_z_m']],cal),float(values[i])
