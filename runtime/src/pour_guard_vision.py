"""RGB template localization and cup-rim normalization; no live object state."""
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image


def project(world,cal):
    T=np.asarray(cal['T']);K=np.asarray(cal['K']);v=(np.asarray(world)-T[:3,3])@T[:3,:3]
    return np.array([K[0,0]*v[0]/(-v[2])+K[0,2],K[1,2]-K[1,1]*v[1]/(-v[2])])


def plane_point(uv,z,cal):
    T=np.asarray(cal['T']);K=np.asarray(cal['K']);u,v=uv
    ray=T[:3,:3]@np.array([(u-K[0,2])/K[0,0],-(v-K[1,2])/K[1,1],-1.])
    return T[:3,3]+ray*((z-T[2,3])/ray[2])


def bottle_base_from_neck(uv,cal):
    neck=plane_point(uv,cal['table_z']+.135,cal)
    return neck-np.array([0.,0.,.135])


def cup_crop(rgb,xy,cal):
    center=np.r_[xy,cal['table_z']]
    rim=project(center+[0,0,.038],cal);base=project(center+[0,0,.004],cal)
    right=np.asarray(cal['T'])[:3,0]
    radius=abs(project(center+[0,0,.038]+right*.024,cal)[0]-rim[0])
    sx=radius/32.;shear=(base[0]-rim[0])/40.;sy=(base[1]-rim[1])/40.
    affine=(sx,shear,rim[0]-48*sx-24*shear,0.,sy,rim[1]-24*sy)
    a=np.asarray(Image.fromarray(rgb).transform((96,96),Image.Transform.AFFINE,affine,Image.Resampling.BILINEAR)).copy()
    a[86:]=180
    return a


def gradients(image):
    if image.ndim==3:image=image[None]
    a=image.float().permute(0,3,1,2)/255.
    gray=(a*torch.tensor([.299,.587,.114],device=a.device)[None,:,None,None]).sum(1,keepdim=True)
    return torch.cat([F.pad((gray[:,:,:,2:]-gray[:,:,:,:-2])*.5,(1,1,0,0)),
        F.pad((gray[:,:,2:,:]-gray[:,:,:-2,:])*.5,(0,0,1,1))],1)


def make_bank(images,anchors,kind,device='cuda',aggregation='median',absolute_gradients=False,anchor_radius_px=None,anchor_penalty=0.):
    # Only the neck outline or glass rim is matched, not liquid volume.
    width,height,dx,dy=(48,64,-24,-12) if kind=='source' else (80,28,-40,-14)
    feat=gradients(torch.as_tensor(np.stack(images),device=device));n=len(images)
    if absolute_gradients:feat=feat.abs()
    xx,yy=torch.meshgrid(torch.arange(width,device=device),torch.arange(height,device=device),indexing='xy')
    uv=torch.tensor(np.asarray(anchors),device=device,dtype=torch.float32)
    grid=torch.stack([(xx[None]+uv[:,0,None,None]+dx)/383*2-1,
                      (yy[None]+uv[:,1,None,None]+dy)/383*2-1],-1)
    kernels=F.grid_sample(feat,grid,align_corners=True)
    mask=torch.ones((1,2,height,width),device=device)
    # The patch center contains the actual bottle/rim; suppress background edges.
    if kind=='source':
        mask[:,:,:,0:5]=0;mask[:,:,:,-5:]=0;mask[:,:,:8]=0;mask[:,:,45:]=0
        mask[:,:,8:19]*=2.
    kernels*=mask;kernels/=kernels.square().sum((1,2,3),keepdim=True).sqrt().clamp(min=1e-6)
    kernels*=mask  # numerator weights both observations; denominator uses mask²
    lo=np.floor(np.min(anchors,axis=0)-9).astype(int);hi=np.ceil(np.max(anchors,axis=0)+9).astype(int)
    return dict(kernels=kernels.cpu(),mask=mask.cpu(),offset=[dx,dy],bounds=[*lo.tolist(),*hi.tolist()],kind=kind,
        aggregation=aggregation,absolute_gradients=absolute_gradients,anchors=np.asarray(anchors).tolist(),anchor_radius_px=anchor_radius_px,anchor_penalty=anchor_penalty)


@torch.inference_mode()
def locate(image,bank,tracking=False,device='cuda',expected_uv=None):
    if bank.get('alternatives'):
        # Each family retains robust median matching; keep older appearance coverage.
        matches=[locate(image,b,tracking,device,expected_uv) for b in bank['alternatives']]
        if bank.get('initial_score_maps') and not tracking and expected_uv is None:
            maps=bank['initial_score_maps'];assert len(maps)==len(matches)
            matches=[(uv,m['offset']+m['scale']*(score-m['anchor'])) for (uv,score),m in zip(matches,maps)]
        return max(matches,key=lambda x:x[1])
    a=torch.as_tensor(image,device=device);feat=gradients(a)
    if bank.get('absolute_gradients'):feat=feat.abs()
    kernels=bank['kernels'].to(device);mask=bank['mask'].to(device);h,w=kernels.shape[-2:];dx,dy=bank['offset']
    x0,y0,x1,y1=bank['bounds']
    if tracking:y0=max(14,y0-95);x0=max(24,x0-12);x1=min(355,x1+12)
    if expected_uv is not None:
        x0,y0=np.floor(np.asarray(expected_uv)-12).astype(int);x1,y1=np.ceil(np.asarray(expected_uv)+12).astype(int)
    startx=max(0,x0+dx);starty=max(0,y0+dy);endx=min(384,x1+dx+w);endy=min(384,y1+dy+h)
    region=feat[:,:,starty:endy,startx:endx]
    numerator=F.conv2d(region,kernels)
    norm=F.conv2d(region.square(),mask.square()).clamp(min=1e-10).sqrt()
    scores=numerator/norm
    if bank.get('anchor_radius_px') is not None and not tracking and expected_uv is None:
        # A template's view-specific background is valid only near its labeled
        # training view. This restriction uses stored calibration, not live poses.
        hh,ww=scores.shape[-2:]
        xx,yy=torch.meshgrid(torch.arange(ww,device=device),torch.arange(hh,device=device),indexing='xy')
        anchors=torch.as_tensor(bank['anchors'],device=device)
        distance=(xx[None]+startx-dx-anchors[:,0,None,None]).square()+(yy[None]+starty-dy-anchors[:,1,None,None]).square()
        scores=scores-bank.get('anchor_penalty',0.)*distance[None]/bank['anchor_radius_px']**2
        scores=scores.masked_fill(distance[None]>bank['anchor_radius_px']**2,-1.)
    aggregation=bank.get('aggregation','median')
    if aggregation=='median':field=scores.median(1).values[0]
    elif aggregation=='max':field=scores.max(1).values[0]
    elif aggregation=='top3':field=scores.topk(min(3,len(kernels)),dim=1).values.mean(1)[0]
    else:raise ValueError(aggregation)
    hh,ww=field.shape;flat=int(field.argmax());y,x=divmod(flat,ww);sub=[]
    for axis,index,limit in ((1,x,ww),(0,y,hh)):
        if 0<index<limit-1:
            left=field[y,x-1] if axis==1 else field[y-1,x];right=field[y,x+1] if axis==1 else field[y+1,x]
            den=float(left-2*field[y,x]+right)
            sub.append(float(np.clip(.5*float(left-right)/den,-.5,.5)) if abs(den)>1e-8 else 0.)
        else:sub.append(0.)
    uv=np.array([startx+x-dx+sub[0],starty+y-dy+sub[1]])
    return uv,float(field[y,x])


def surface_features(image):
    a=image.float()/255.;r,g,b=a.unbind(-1)
    mask=((r>1.4*g+.025)&(r>1.2*b+.02)).float()
    mask=mask[:,:,35:73].clone();mask[:,:14]=0;mask[:,83:]=0
    hist=mask.mean(2);mass=hist.sum(1).clamp(min=1e-6)
    yy=torch.arange(96,device=a.device)[None]/95.
    mean=(hist*yy).sum(1)/mass
    std=((hist*(yy-mean[:,None]).square()).sum(1)/mass).sqrt()
    features=torch.stack([torch.ones_like(mean),mean,mean.square(),std],1)
    return features,hist,mass
