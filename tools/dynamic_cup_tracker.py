"""Stationary-cup visual tracking; no live simulator pose or fill input."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from pour_cup_locator import CupLocator,centers
from pour_guard_vision import project,plane_point
from pour_temporal_localization import consensus
from scale_rim_refiner import RimRefiner,coordinates,crop_native,correct_rim

class CupTracker:
    def __init__(self,artifact,geometry,device='cuda'):
        self.artifact=artifact;self.geometry=geometry;self.device=device
        self.coarse=CupLocator().to(device).eval();self.coarse.load_state_dict(artifact['coarse_state'])
        self.fine=[]
        for bank in artifact['fine_banks']:
            model=RimRefiner().to(device).eval();model.load_state_dict(bank['state']);self.fine.append(model)
        self.history=[];self.anchor=None;self.template=None;self.failed=False

    @torch.inference_mode()
    def detect(self,image,refine=True):
        small=np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy()
        uv,score=centers(self.coarse(torch.as_tensor(small,device=self.device)[None]));center=uv[0].cpu().numpy()*(1024/384);score=float(score[0])
        if not refine:return center,score
        crops=[];origins=[]
        for dx,dy in self.artifact['settings']['shifts']:
            crop,origin=crop_native(image,center+np.array([dx,dy]));crops.append(crop);origins.append(origin)
        batch=torch.as_tensor(np.stack(crops),device=self.device);points=[]
        for model,bank in zip(self.fine,self.artifact['fine_banks']):
            predicted=coordinates(model(batch)).cpu().numpy()+np.asarray(origins)
            values=[correct_rim(p,score,bank) for p in predicted]
            if min(s for p,s in values)<=0:score=0.
            points.append(np.mean([p for p,s in values],axis=0))
        point=np.mean(points,axis=0)
        if np.linalg.norm(point-center)>self.artifact['settings']['max_refinement_pixels']:score=0.
        return point,score

    def box(self):
        cal=self.artifact['calibration'];g=self.geometry
        xy=plane_point(self.anchor,cal['table_z']+g['rim_z_m'],cal)[:2]
        base=np.r_[xy,cal['table_z']+g['bottom_z_m']];uv=project(base,cal)
        radius=g['profile'][0][1]+.0024;right=np.asarray(cal['T'])[:3,0]
        rx=abs(project(base+right*radius,cal)[0]-uv[0])
        # The lower outline contains stable glass/base edges, away from the bottle mouth.
        lo,hi=self.artifact['settings'].get('reference_vertical_fraction',[-.30,.42])
        return np.rint([uv[0]-1.12*rx,uv[1]+lo*rx,uv[0]+1.12*rx,uv[1]+hi*rx]).astype(int)

    def features(self,image):
        rgb=torch.as_tensor(np.asarray(image).copy(),device=self.device,dtype=torch.float32)
        gray=(rgb*rgb.new_tensor([.299,.587,.114])).sum(-1)[None,None]/255
        gray=F.avg_pool2d(gray,3,1,1)
        return torch.cat((F.pad((gray[:,:,:,2:]-gray[:,:,:,:-2])*.5,(1,1,0,0)),F.pad((gray[:,:,2:,:]-gray[:,:,:-2,:])*.5,(0,0,1,1))),1)

    def establish(self,image):
        self.bounds=self.box();x0,y0,x1,y1=self.bounds;m=self.artifact['settings']['search_pixels']
        if min(x0-m,y0-m)<0 or max(x1+m,y1+m)>1024:return False
        feature=self.features(image);self.template=feature[:,:,y0:y1,x0:x1].clone()
        return float(self.template.square().sum())>1e-6

    @torch.inference_mode()
    def match(self,image):
        x0,y0,x1,y1=self.bounds;m=self.artifact['settings']['search_pixels']
        region=self.features(image)[:,:,y0-m:y1+m,x0-m:x1+m]
        cal=self.artifact['calibration'];z=cal['table_z']+self.geometry['bottom_z_m'];mid=(self.bounds[:2]+self.bounds[2:])/2
        parts=self.artifact['settings'].get('template_parts',1);edges=np.rint(np.linspace(0,self.template.shape[-1],parts+1)).astype(int);matches=[]
        for lo,hi in zip(edges,edges[1:]):
            template=self.template[:,:,:,lo:hi];patch=region[:,:,:,lo:hi+2*m]
            numerator=F.conv2d(patch,template)[0,0]
            norm=F.conv2d(patch.square(),torch.ones_like(template))[0,0]*template.square().sum()
            response=numerator/norm.clamp(min=1e-12).sqrt();index=int(response.argmax());dy,dx=divmod(index,response.shape[1]);delta=np.array([dx-m,dy-m],float)
            movement=float(np.linalg.norm(plane_point(mid+delta,z,cal)[:2]-plane_point(mid,z,cal)[:2])*1000)
            matches.append(dict(similarity=float(response.flatten()[index]),movement_mm=movement,shift_pixels=delta.tolist()))
        self.last_matches=matches
        if parts==1:
            v=matches[0];return v['similarity'],v['movement_mm'],v['shift_pixels']
        good=[v for v in matches if v['similarity']>=self.artifact['settings']['minimum_similarity'] and v['movement_mm']<=self.artifact['settings']['maximum_movement_mm']]
        if len(good)<self.artifact['settings']['minimum_support']:return 0.,max(v['movement_mm'] for v in matches),[0.,0.]
        return min(v['similarity'] for v in good),max(v['movement_mm'] for v in good),np.median([v['shift_pixels'] for v in good],axis=0).tolist()

    @torch.inference_mode()
    def update(self,image):
        assert np.asarray(image).shape==(1024,1024,3)
        if self.failed:return dict(valid=False,reason='latched_rejection',uv=None)
        initial=self.anchor is None;point,score=self.detect(image,refine=initial)
        out=dict(valid=False,reason='confirming',uv=None,score=score,raw_uv=point.tolist() if initial else None)
        if not np.isfinite(point).all() or score<self.artifact['threshold']:
            self.failed=True;out['reason']='cup_absent_or_uncertain';return out
        if initial:
            self.history.append((point.tolist(),score));confirmed=consensus(self.history,self.artifact['threshold'],self.artifact['consensus'])
            if confirmed is None:
                if len(self.history)>=12:self.failed=True;out['reason']='initial_confirmation_failed'
                return out
            self.anchor=np.asarray(confirmed['uv'])
            if not self.establish(image):self.failed=True;out['reason']='reference_unusable';return out
        else:
            if np.linalg.norm(point-self.anchor)>self.artifact['settings']['association_pixels']:
                self.failed=True;out['reason']='cup_association_lost';return out
            similarity,movement,shift=self.match(image);out.update(similarity=similarity,movement_mm=movement,shift_pixels=shift)
            if hasattr(self,'last_matches'):out['patch_matches']=self.last_matches
            if similarity<self.artifact['settings']['minimum_similarity'] or movement>self.artifact['settings']['maximum_movement_mm']:
                self.failed=True;out['reason']='cup_moved_or_reference_changed';return out
        out.update(valid=True,reason='tracked',uv=self.anchor.tolist());return out
