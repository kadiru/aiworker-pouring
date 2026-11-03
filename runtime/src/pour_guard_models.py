"""Cup height model using a visually normalized vessel crop."""
import torch
from torch import nn
from pour_stop_models import VisualEncoder
from pour_guard_vision import surface_features


class CupEstimator(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=VisualEncoder()
        self.head=nn.Sequential(nn.Linear(164,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,image):
        # Bottle/arm pixels to the left are not evidence of the cup level.
        image=image.clone();image[:,:,:30]=180;image[:,:,82:]=180;image[:,86:]=180
        features,hist,mass=surface_features(image)
        latent=self.encoder(image)
        return self.head(torch.cat([latent,hist,features],1))[:,0].sigmoid()
