"""Supervised RGB rim-center heatmap and subpixel offsets."""
import numpy as np
import torch
from torch import nn

class CupLocator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),nn.ReLU(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=4,dilation=4),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=8,dilation=8),nn.ReLU(),nn.Conv2d(32,3,1))
        with torch.no_grad():self.net[-1].bias[0]=-4.4
    def forward(self,image):return self.net(image.float().permute(0,3,1,2)/255.)

def centers(output):
    batch,_,height,width=output.shape
    score,flat=output[:,0].sigmoid().flatten(1).max(1)
    y=flat//width;x=flat%width;ids=torch.arange(batch,device=output.device)
    offsets=output[ids,1:,y,x].sigmoid()
    return (torch.stack((x,y),1)+offsets)*4,score

_CACHE={}
def combine_predictions(predictions,max_disagreement_px):
    uv=np.asarray([p[0] for p in predictions]);score=min(p[1] for p in predictions)
    if np.linalg.norm(uv[0]-uv[1])>max_disagreement_px:score=0.
    return uv.mean(0),float(score)

def correct_prediction(prediction,bank):
    if 'affine_uv_correction' not in bank:return prediction
    uv,score=prediction;uv=np.asarray(uv)
    delta=np.r_[1.,(uv-192.)/192.]@np.asarray(bank['affine_uv_correction'])
    if np.linalg.norm(delta)>bank['max_correction_px']:score=0.
    return uv+delta,float(score)

@torch.inference_mode()
def locate_cup(image,bank,device='cuda'):
    if bank['kind']=='cup_heatmap_ensemble':
        assert len(bank['members'])==2
        result=combine_predictions([locate_cup(image,b,device) for b in bank['members']],bank['max_disagreement_px'])
        return correct_prediction(result,bank)
    key=(bank['identity'],str(device))
    if key not in _CACHE:
        model=CupLocator().to(device).eval();model.load_state_dict(bank['state']);_CACHE[key]=model
    uv,score=centers(_CACHE[key](torch.as_tensor(np.asarray(image).copy(),device=device)[None]))
    return correct_prediction((uv[0].cpu().numpy(),float(score[0])),bank)
