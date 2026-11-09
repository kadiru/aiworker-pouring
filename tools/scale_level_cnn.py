"""Size-aware trainable level CNN, isolated from the active robot model."""
import torch
from torch import nn
from pour_guard_models import CupEstimator
from scale_level_readout import geometry_tensor

class ScaleLevelCNN(nn.Module):
    def __init__(self):
        super().__init__();self.cup=CupEstimator()
        self.correction=nn.Sequential(nn.Linear(11,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
    def forward(self,image,context):
        level=self.cup(image);a=torch.cat((context,level[:,None],context*level[:,None]),1)
        return (level+self.correction(a)[:,0]).clamp(0,1)

def level_context(image,geometry,xy):
    dims=geometry_tensor(geometry,image).float()
    pose=torch.as_tensor(xy,device=image.device,dtype=torch.float32).reshape(-1,2)
    if len(pose)==1:pose=pose.expand(len(image),2)
    return torch.cat(((dims-1)*5,(pose-pose.new_tensor([.5,-.225]))/.15),1)

_CACHE={}
@torch.inference_mode()
def predict_size_cnn(image,xy,geometry,artifact):
    key=(artifact['size_cnn_identity'],str(image.device))
    if key not in _CACHE:
        model=ScaleLevelCNN().to(image.device).eval();model.load_state_dict(artifact['size_cnn_state']);_CACHE[key]=model
    return _CACHE[key](image,level_context(image,geometry,xy))
