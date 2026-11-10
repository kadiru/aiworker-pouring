"""Separate color-mask network: no primary RGB weights or predictions as inputs."""
import torch
from scale_level_cnn import ScaleLevelCNN,level_context

def mask_image(image):
    rgb=image.to(torch.int64);r,g,b=rgb.unbind(-1)
    mask=(r*1000>g*1400+6375)&(r*1000>b*1200+5100)&(r>30)
    out=torch.full_like(image,180)
    out[mask]=image.new_tensor([255,0,0])
    valid=mask[:,14:83,30:82].sum((1,2))>=20
    return out,valid

_CACHE={}
@torch.inference_mode()
def predict_mask(image,xy,geometry,artifact):
    key=(artifact['identity'],str(image.device))
    if key not in _CACHE:
        model=ScaleLevelCNN().to(image.device).eval();model.load_state_dict(artifact['state']);_CACHE[key]=model
    x,valid=mask_image(image);value=_CACHE[key](x,level_context(x,geometry,xy))
    return torch.where(valid,value,torch.full_like(value,float('nan'))),valid
