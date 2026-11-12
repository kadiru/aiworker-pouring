"""RGB template localization and cup-rim normalization; no live object state."""
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from pour_bottle_geometry import BOTTLE


def project(world,cal):
    T=np.asarray(cal['T']);K=np.asarray(cal['K']);v=(np.asarray(world)-T[:3,3])@T[:3,:3]
    return np.array([K[0,0]*v[0]/(-v[2])+K[0,2],K[1,2]-K[1,1]*v[1]/(-v[2])])


def plane_point(uv,z,cal):
    T=np.asarray(cal['T']);K=np.asarray(cal['K']);u,v=uv
    ray=T[:3,:3]@np.array([(u-K[0,2])/K[0,0],-(v-K[1,2])/K[1,1],-1.])
    return T[:3,3]+ray*((z-T[2,3])/ray[2])


def bottle_base_from_neck(uv,cal,lip_height=BOTTLE[-1][0]):
    neck=plane_point(uv,cal['table_z']+lip_height,cal)
    return neck-np.array([0.,0.,lip_height])


def cup_crop(rgb,xy,cal):
    geometry=cal.get('cup_geometry',{})
    rim_z=geometry.get('rim_z_m',.038);bottom_z=geometry.get('bottom_z_m',.004)
    rim_radius=geometry.get('rim_radius_m',.024)
    center=np.r_[xy,cal['table_z']]
    rim=project(center+[0,0,rim_z],cal);base=project(center+[0,0,bottom_z],cal)
    right=np.asarray(cal['T'])[:3,0]
    radius=abs(project(center+[0,0,rim_z]+right*rim_radius,cal)[0]-rim[0])
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


def make_bank(images,anchors,kind,device='cuda',aggregation='median',absolute_gradients=False,anchor_radius_px=None,anchor_penalty=0.,source_mask_box=None,gradient_clip=None,cup_mask_box=None):
    # Only the neck outline or glass rim is matched, not liquid volume.
    width,height,dx,dy=(48,64,-24,-12) if kind=='source' else (80,28,-40,-14)
    feat=gradients(torch.as_tensor(np.stack(images),device=device));n=len(images)
    if gradient_clip is not None:feat=feat.clamp(-gradient_clip,gradient_clip)
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
    if source_mask_box is not None:
        assert kind=='source' and len(source_mask_box)==4
        x0,y0,x1,y1=source_mask_box
        assert 0<=x0<x1<=width and 0<=y0<y1<=height
        keep=torch.zeros_like(mask);keep[:,:,y0:y1,x0:x1]=1
        mask*=keep
    if cup_mask_box is not None:
        assert kind=='cup' and len(cup_mask_box)==4
        x0,y0,x1,y1=cup_mask_box
        assert 0<=x0<x1<=width and 0<=y0<y1<=height
        keep=torch.zeros_like(mask);keep[:,:,y0:y1,x0:x1]=1
        mask*=keep
    kernels*=mask;kernels/=kernels.square().sum((1,2,3),keepdim=True).sqrt().clamp(min=1e-6)
    kernels*=mask  # numerator weights both observations; denominator uses mask²
    lo=np.floor(np.min(anchors,axis=0)-9).astype(int);hi=np.ceil(np.max(anchors,axis=0)+9).astype(int)
    return dict(kernels=kernels.cpu(),mask=mask.cpu(),offset=[dx,dy],bounds=[*lo.tolist(),*hi.tolist()],kind=kind,
        aggregation=aggregation,absolute_gradients=absolute_gradients,anchors=np.asarray(anchors).tolist(),anchor_radius_px=anchor_radius_px,anchor_penalty=anchor_penalty,
        source_mask_box=source_mask_box,gradient_clip=gradient_clip,cup_mask_box=cup_mask_box)


@torch.inference_mode()
def locate(image,bank,tracking=False,device='cuda',expected_uv=None):
    if bank.get('kind') in ('cup_heatmap','cup_heatmap_ensemble'):
        from pour_cup_locator import locate_cup
        return locate_cup(image,bank,device)
    if bank.get('alternatives'):
        # Each family retains robust median matching; keep older appearance coverage.
        matches=[locate(image,b,tracking,device,expected_uv) for b in bank['alternatives']]
        if bank.get('initial_score_maps') and not tracking and expected_uv is None:
            maps=bank['initial_score_maps'];assert len(maps)==len(matches)
            matches=[(uv,m['offset']+m['scale']*(score-m['anchor'])) for (uv,score),m in zip(matches,maps)]
        return max(matches,key=lambda x:x[1])
    a=torch.as_tensor(image,device=device);feat=gradients(a)
    if bank.get('gradient_clip') is not None:feat=feat.clamp(-bank['gradient_clip'],bank['gradient_clip'])
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
    if bank.get('initial_region_uv') is not None and not tracking and expected_uv is None:
        # A calibrated tabletop support prior learned from training positions.
        # Background edges above/outside that workspace are not object anchors.
        hh,ww=field.shape
        xx,yy=torch.meshgrid(torch.arange(ww,device=device),torch.arange(hh,device=device),indexing='xy')
        uv=torch.stack((xx+startx-dx,yy+starty-dy),-1)
        polygon=torch.as_tensor(bank['initial_region_uv'],device=device,dtype=field.dtype)
        edge=polygon.roll(-1,0)-polygon;relative=uv[:,:,None]-polygon
        cross=edge[:,0]*relative[:,:,:,1]-edge[:,1]*relative[:,:,:,0]
        inside=(cross>=-1e-5).all(-1)|(cross<=1e-5).all(-1)
        field=field.masked_fill(~inside,-1.)
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


def color_geometry_features(image,xy=None):
    """Spatial color-envelope features independent of the level CNN."""
    base,hist,mass=surface_features(image)
    qs=image.new_tensor([.02,.05,.1,.25,.5,.75,.9,.95,.98],dtype=torch.float32)
    cumulative=hist.cumsum(1)/hist.sum(1,keepdim=True).clamp(min=1e-6)
    quantiles=(cumulative[:,:,None]>=qs).float().argmax(1).float()/95.
    rgb=image.float()/255.;r,g,b=rgb.unbind(-1)
    red=((r>1.4*g+.025)&(r>1.2*b+.02)).float();red[:,:14]=0;red[:,83:]=0
    side=[];yy=torch.arange(96,device=image.device)[None]/95.
    for low,high in ((35,48),(48,61),(61,74)):
        rows=red[:,:,low:high].mean(2);total=rows.sum(1).clamp(min=1e-6)
        center=(rows*yy).sum(1)/total
        top=(rows.cumsum(1)/total[:,None]>=.05).float().argmax(1).float()/95.
        side.extend((center,top))
    features=torch.cat((base,quantiles,torch.stack(side,1)),1)
    if xy is None:return features
    position=torch.as_tensor(xy,device=image.device,dtype=torch.float32).reshape(-1,2)
    if len(position)==1:position=position.expand(len(image),2)
    assert len(position)==len(image)
    position=(position-position.new_tensor([.5,-.225]))/.15
    interactions=(features[:,[1,5,8],None]*position[:,None,:]).flatten(1)
    return torch.cat((features,position,interactions),1)


def color_geometry_readout(image,fitted,xy=None):
    if fitted.get('feature_kind')=='color_kernel_v3':
        use_pose=fitted['dimensions']==27
        if use_pose:assert xy is not None,'Image-localized cup position is required'
        x=color_geometry_features(image,xy if use_pose else None)[:,:fitted['dimensions']].double()
        x=(x-x.new_tensor(fitted['mean']))/x.new_tensor(fitted['scale'])
        centers=x.new_tensor(fitted['centers']);distance=(x[:,None]-centers[None]).square().mean(-1)
        kernel=torch.exp(-distance/(2*fitted['length_scale']**2))
        return (kernel@x.new_tensor(fitted['alpha'])+fitted['bias']).clamp(0,1)
    if fitted.get('feature_kind') in ('color_envelope_v1','color_envelope_pose_v2'):
        use_pose=fitted['feature_kind']=='color_envelope_pose_v2'
        if use_pose:assert xy is not None,'Image-localized cup position is required'
        x=color_geometry_features(image,xy if use_pose else None)[:,:fitted['dimensions']]
        x=(x-x.new_tensor(fitted['mean']))/x.new_tensor(fitted['scale'])
    else:x=surface_features(image)[0][:,:fitted['dimensions']]
    return (x@x.new_tensor(fitted['coef'])+fitted.get('bias',0.)).clamp(0,1)
