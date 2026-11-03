"""Separate image-only liquid estimator and pouring-increment BC readout."""
import torch
from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,32,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.ReLU())

    def forward(self,x):return self.net(x.permute(0,3,1,2).float()/255.)


class LiquidDetector(nn.Module):
    def __init__(self):
        super().__init__();self.cup=VisualEncoder();self.source=VisualEncoder();self.wrist=VisualEncoder()
        self.cup_head=nn.Linear(164,2);self.source_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))

    def forward(self,cup,source,wrist):
        x=cup.float()/255.;r,g,b=x.unbind(-1)
        red=((r>1.4*g+.025)&(r>1.2*b+.02)).float()
        yy=torch.linspace(0,1,x.shape[1],device=x.device)[None,:,None]
        xx=torch.linspace(0,1,x.shape[2],device=x.device)[None,None,:]
        mass=red.sum((1,2)).clamp(min=1)
        color=torch.stack([red.mean((1,2)),(red*yy).sum((1,2))/mass,(red*xx).sum((1,2))/mass,red.mean(2).max(1).values],-1)
        # Give the estimator explicit spatial evidence: red below/outside the
        # vessel must not be treated as a high liquid surface. Fixed camera rig.
        central=red[:,:,28:68].mean(2)
        central[:,:28]=0;central[:,78:]=0
        c=self.cup_head(torch.cat([self.cup(cup),color,central],-1))
        s=self.source_head(torch.cat([self.source(source),self.wrist(wrist)],-1))
        return dict(fill=c[:,0].sigmoid(),full_logit=c[:,1],left=s[:,0].sigmoid(),empty_logit=s[:,1],present_logit=s[:,2])


class PourIncrement(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(223,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,7))

    def forward(self,x):return self.net(x)
