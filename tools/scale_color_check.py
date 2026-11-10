"""Independent color/geometry features. Never takes a CNN prediction as input."""
import torch
from pour_guard_vision import color_geometry_features
from scale_level_readout import geometry_tensor,predict_readout
from scale_surface_geometry import envelope_features

def color_features(image,xy,geometry):
    visual=color_geometry_features(image,xy).double();dims=geometry_tensor(geometry,image)
    core=visual[:,1:19]
    features=torch.cat((visual,dims,core*(dims[:,:1]-1),core*(dims[:,1:2]-1)),1)
    rgb=image.to(torch.int64);r,g,b=rgb.unbind(-1)
    mask=(r*1000>g*1400+6375)&(r*1000>b*1200+5100)&(r>30)
    visible=mask[:,14:83,22:74].sum((1,2))>=20
    return features,visible

def predict_color(image,xy,geometry,artifact,surface_image=None,lookup=None):
    fitted=artifact['color_readout']
    x,valid=feature_matrix(image,xy,geometry,fitted['features'],surface_image,lookup)
    value=predict_readout(x,fitted)
    return torch.where(valid,value,torch.full_like(value,float('nan'))),valid

def feature_matrix(image,xy,geometry,kind,surface_image=None,lookup=None):
    if kind in ('color_geometry','color_compact'):
        x,valid=color_features(image,xy,geometry)
        if kind=='color_compact':x=torch.cat((x[:,1:13],x[:,19:24],x[:,27:30]),1)
        return x,valid
    if surface_image is None or lookup is None:raise ValueError('Native surface crop/lookup required')
    x,_,valid=envelope_features(surface_image,lookup,geometry,xy)
    if kind=='physical_compact':x=torch.cat((x[:,39:42],x[:,94:]),1)
    elif kind=='physical_linear':x=torch.cat((x[:,39:42],x[:,94:99]),1)
    elif kind!='physical_surface':raise ValueError(kind)
    return x,valid
