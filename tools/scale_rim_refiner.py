"""Study-local native-resolution rim refinement; never reads object poses."""
import numpy as np
from PIL import Image
import torch
from torch import nn
from pour_cup_locator import CupLocator, centers

class RimRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=2,dilation=2),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=4,dilation=4),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=8,dilation=8),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=16,dilation=16),nn.ReLU(),nn.Conv2d(32,1,1))
    def forward(self,x): return self.net(x.float().permute(0,3,1,2)/255.)[:,0]

def coordinates(logits):
    h,w=logits.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=logits.device,dtype=logits.dtype)*4+2,
                       torch.arange(w,device=logits.device,dtype=logits.dtype)*4+2,indexing='ij')
    p=logits.flatten(1).softmax(1).reshape_as(logits)
    return torch.stack(((p*x).sum((1,2)),(p*y).sum((1,2))),1)

def crop_native(image,center,size=256):
    """Integer-origin crop with explicit padding and reversible UV coordinates."""
    image=np.asarray(image); origin=np.rint(center).astype(int)-size//2
    x,y=origin; h,w=image.shape[:2]; out=np.full((size,size,3),180,dtype=np.uint8)
    x0,y0=max(0,x),max(0,y); x1,y1=min(w,x+size),min(h,y+size)
    if x1>x0 and y1>y0:out[y0-y:y1-y,x0-x:x1-x]=image[y0:y1,x0:x1]
    return out,origin

def native_coarse(image,model,device):
    small=np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.BILINEAR)).copy()
    uv,score=centers(model(torch.as_tensor(small,device=device)[None]))
    return uv[0].detach().cpu().numpy()*(1024/384),float(score[0])

_CACHE={}
def correct_rim(point,score,bank):
    point=np.asarray(point)
    if 'affine_uv_correction' not in bank:return point,float(score)
    delta=np.r_[1.,(point-512.)/512.]@np.asarray(bank['affine_uv_correction'])
    if not np.isfinite(delta).all() or np.linalg.norm(delta)>bank['max_affine_correction_pixels']:score=0.
    return point+delta,float(score)

@torch.inference_mode()
def locate_static(image,bank,device='cuda'):
    if bank['kind']!='native_rim_refiner':
        from pour_guard_vision import locate
        return locate(image,bank,device=device)
    assert np.asarray(image).shape==(1024,1024,3)
    key=(bank['identity'],str(device))
    if key not in _CACHE:
        coarse=CupLocator().to(device).eval(); coarse.load_state_dict(bank['coarse_state'])
        fine=RimRefiner().to(device).eval(); fine.load_state_dict(bank['state']); _CACHE[key]=(coarse,fine)
    coarse,fine=_CACHE[key]; center,score=native_coarse(image,coarse,device)
    crop,origin=crop_native(image,center,bank['crop_size'])
    point=coordinates(fine(torch.as_tensor(crop,device=device)[None]))[0].cpu().numpy()+origin
    if not np.isfinite(point).all() or np.linalg.norm(point-center)>bank['max_correction_pixels']:score=0.
    return correct_rim(point,score,bank)
