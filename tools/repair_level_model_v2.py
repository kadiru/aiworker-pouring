"""Second development ablation: use cup geometry, omit absolute table XY."""
import torch
from repair_level_model import OutlineLevelNet,color_outline
from scale_level_cnn import level_context

def repair_context(image,geometry,xy):
    context=level_context(image,geometry,xy)
    # Table placement must not act as a surrogate fill label. Perspective is
    # still present in the image; known cup dimensions remain explicit inputs.
    context[:,3:]=0
    return context

_CACHE={}
@torch.inference_mode()
def predict_repair(image,xy,geometry,artifact):
    key=(artifact['identity'],str(image.device))
    if key not in _CACHE:
        model=OutlineLevelNet().to(image.device).eval();model.load_state_dict(artifact['state']);_CACHE[key]=model
    x=image;valid=torch.ones(len(x),dtype=torch.bool,device=x.device)
    if artifact['branch']=='color':x,valid=color_outline(x)
    value=_CACHE[key](x,repair_context(x,geometry,xy))
    return torch.where(valid,value,torch.full_like(value,float('nan'))),valid

def episode_mae(prediction,data):
    errors=(prediction-data['y']).abs()
    names=sorted({m['name'] for m in data['meta']})
    return {n:float(errors[torch.tensor([m['name']==n for m in data['meta']],device=errors.device)].mean()) for n in names}
