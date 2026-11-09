"""High-resolution color-envelope height using known cup/camera geometry."""
import numpy as np
from PIL import Image
import torch
from pour_guard_vision import project

WIDTH=192
COLUMNS=np.linspace(64,128,13)
LEVELS=np.linspace(0,1,101)

def crop_transform(xy,cal,geometry):
    center=np.r_[xy,cal['table_z']];rim=project(center+[0,0,geometry['rim_z_m']],cal)
    base=project(center+[0,0,geometry['bottom_z_m']],cal);right=np.asarray(cal['T'])[:3,0]
    radius=abs(project(center+[0,0,geometry['rim_z_m']]+right*geometry['rim_radius_m'],cal)[0]-rim[0])
    sx=radius/64.;shear=(base[0]-rim[0])/80.;sy=(base[1]-rim[1])/80.
    return np.array([[sx,shear,rim[0]-96*sx-48*shear],[0,sy,rim[1]-48*sy]])

def surface_crop(rgb,xy,cal,geometry):
    transform=crop_transform(xy,cal,geometry)
    return np.asarray(Image.fromarray(rgb).transform((192,192),Image.Transform.AFFINE,tuple(transform.ravel()),Image.Resampling.BILINEAR)).copy()

def surface_lookup(xy,cal,g):
    """Project horizontal rings over [empty, brim], with no observed level input."""
    transform=crop_transform(xy,cal,g);inverse=np.linalg.inv(transform[:,:2])
    z=g['bottom_z_m']+LEVELS*(g['rim_z_m']-g['bottom_z_m'])
    p=np.asarray(g['profile']);radius=np.interp(z,p[:,0],p[:,1]);theta=np.linspace(0,2*np.pi,256,endpoint=False)
    points=np.stack((xy[0]+radius[:,None]*np.cos(theta),xy[1]+radius[:,None]*np.sin(theta),np.broadcast_to(cal['table_z']+z[:,None],(101,256))),-1)
    t=np.asarray(cal['T']);k=np.asarray(cal['K']);v=(points-t[:3,3])@t[:3,:3]
    uv=np.stack((k[0,0]*v[...,0]/-v[...,2]+k[0,2],k[1,2]-k[1,1]*v[...,1]/-v[...,2]),-1)
    normalized=(uv-transform[:,2])@inverse.T;rows=[]
    for ring in normalized:
        left=int(ring[:,0].argmin());right=int(ring[:,0].argmax())
        ring=np.roll(ring,-left,axis=0);right=(right-left)%len(ring)
        first=ring[:right+1];second=np.r_[ring[right:],ring[:1]][::-1]
        assert first[0,0]<COLUMNS.min() and first[-1,0]>COLUMNS.max()
        rows.append(np.minimum(np.interp(COLUMNS,first[:,0],first[:,1]),np.interp(COLUMNS,second[:,0],second[:,1])))
    lookup=np.asarray(rows).T[:,::-1].copy()
    assert np.isfinite(lookup).all() and (np.diff(lookup,axis=1)>0).all(), 'Surface projection must be monotonic'
    return lookup

def envelope_features(image,lookup,geometry,xy):
    """Exact-integer red mask; interpolated quantiles -> projected ring heights."""
    assert image.shape[1:]==(192,192,3)
    rgb=image.to(torch.int64);r,g,b=rgb.unbind(-1)
    red=(r*1000>g*1400+6375)&(r*1000>b*1200+5100)&(r>30)
    red[:,176:]=False
    columns=[red[:,:,int(round(x))-2:int(round(x))+3].sum(2) for x in COLUMNS]
    counts=torch.stack(columns,1);cumulative=counts.cumsum(2);total=counts.sum(2)
    readings=[];quantiles=[];lut=torch.as_tensor(lookup,device=image.device,dtype=torch.float64).contiguous()
    assert lut.shape==(len(image),13,101)
    for percent in (2,5,10):
        row=(cumulative*100>=total[:,:,None]*percent).to(torch.int64).argmax(2)
        before=torch.gather(cumulative,2,(row-1).clamp(min=0)[:,:,None])[:,:,0];before=torch.where(row>0,before,0)
        mass=torch.gather(counts,2,row[:,:,None])[:,:,0].clamp(min=1)
        y=row.double()+(total.double()*percent/100-before)/mass-.5
        index=torch.searchsorted(lut,y[:,:,None].contiguous()).squeeze(2).clamp(1,100)
        lo=torch.gather(lut,2,(index-1)[:,:,None])[:,:,0];hi=torch.gather(lut,2,index[:,:,None])[:,:,0]
        level=1-((index-1)+(y-lo)/(hi-lo))*.01
        readings.append(level.clamp(-.25,1.25));quantiles.append(y/192.)
    from scale_level_readout import geometry_tensor
    dims=geometry_tensor(geometry,image);pose=torch.as_tensor(xy,device=image.device,dtype=torch.float64).reshape(-1,2)
    if len(pose)==1:pose=pose.expand(len(image),2)
    view=(pose-pose.new_tensor([.5,-.225]))/.15
    physical=torch.cat(readings,1);summary=torch.stack([q.median(1).values for q in readings],1)
    x=torch.cat((physical,summary,torch.cat(quantiles,1),total.double()/880.,dims,view,
        summary*view[:,:1],summary*view[:,1:],summary*(dims[:,:1]-1),summary*(dims[:,1:2]-1),summary*(dims[:,2:3]-1)),1)
    valid=(total>=10).all(1)
    return x,summary[:,1],valid
