"""Continuous pixel translation with matching supervised rim coordinates."""
import torch
from torch.nn import functional as F


def translate(images,uv,dx,dy):
    n,h,w,c=images.shape
    assert c==3 and uv.shape==(n,2)
    x=images.float().permute(0,3,1,2)-180.
    theta=x.new_tensor([[1.,0.,-2*dx/w],[0.,1.,-2*dy/h]])[None].expand(n,-1,-1)
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    shifted=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)+180.
    return shifted.permute(0,2,3,1).contiguous(),uv+uv.new_tensor([dx,dy])
