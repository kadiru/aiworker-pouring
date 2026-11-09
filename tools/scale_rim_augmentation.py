"""Image/label-consistent affine crop augmentation in native pixel units."""
import torch
from torch.nn import functional as F

def affine(image,points,matrix,shift):
    """Forward point map q=A(p-c)+c+t; inverse map resamples RGB.

    Pixel centers are integer coordinates; align_corners=False is explicit.
    """
    n,h,w,_=image.shape;device=image.device;dtype=image.dtype
    center=image.new_tensor([(w-1)/2,(h-1)/2])
    y,x=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(w,device=device,dtype=dtype),indexing='ij')
    destination=torch.stack((x,y),-1).reshape(1,-1,2)
    inverse=torch.linalg.inv(matrix)
    source=torch.bmm(destination-center-shift[:,None,:],inverse.transpose(1,2))+center
    normalized=(source+.5)/image.new_tensor([w,h])*2-1
    transformed=F.grid_sample(image.permute(0,3,1,2)-180,normalized.reshape(n,h,w,2),
                              mode='bilinear',padding_mode='zeros',align_corners=False)+180
    targets=torch.bmm((points-center)[:,None,:],matrix.transpose(1,2))[:,0]+center+shift
    return transformed.permute(0,2,3,1).contiguous(),targets
