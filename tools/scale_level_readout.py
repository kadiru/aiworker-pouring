"""Study-only size-aware level prediction from RGB and declared cup geometry."""
import torch
from pour_settled_model import features,predict

def geometry_tensor(geometry,image):
    values=geometry if isinstance(geometry,list) else [geometry]*len(image)
    assert len(values)==len(image)
    a=image.new_tensor([[(g['rim_z_m']-g['bottom_z_m'])/.034,g['rim_radius_m']/.024] for g in values],dtype=torch.float64)
    if not torch.isfinite(a).all() or not ((a>=.89)&(a<=1.11)).all():raise ValueError('Unsupported declared cup geometry')
    return torch.cat((a,a[:,1:2]/a[:,:1]),1)

def level_features(image,cnn,xy,geometry,kind):
    base=features(image,cnn,xy).double();dims=geometry_tensor(geometry,image)
    view=base[:,5:7]*.08
    # Exact-count quantiles, color moments, CNN and global continuous color shape.
    visual=torch.cat((base[:,1:5],base[:,340:349],base[:,-2:]),1)
    compact=torch.cat((visual,view,dims,visual*view[:,:1],visual*view[:,1:],
        visual*(dims[:,:1]-1),visual*(dims[:,1:2]-1),visual*(dims[:,2:3]-1)),1)
    if kind=='compact_geometry':return compact
    if kind=='spatial_geometry':return torch.cat((compact,base[:,604:-2]),1)
    raise ValueError(kind)

def predict_readout(x,fitted):
    z=(x.double()-x.new_tensor(fitted['mean']))/x.new_tensor(fitted['scale'])
    if fitted['kind']=='ridge':
        return (z@x.new_tensor(fitted['coef'])+fitted['bias']).clamp(0,1)
    if fitted['kind']=='rbf':
        centers=x.new_tensor(fitted['centers'])
        distance=(z[:,None]-centers[None]).square().mean(-1)
        return (torch.exp(-distance/(2*fitted['length_scale']**2))@x.new_tensor(fitted['alpha'])+fitted['bias']).clamp(0,1)
    raise ValueError(fitted['kind'])

def predict_static_level(image,cnn,xy,geometry,artifact):
    if 'size_cnn_state' in artifact:
        from scale_level_cnn import predict_size_cnn
        return predict_size_cnn(image,xy,geometry,artifact)
    if 'size_level_readout' in artifact:
        fitted=artifact['size_level_readout']
        return predict_readout(level_features(image,cnn,xy,geometry,fitted['features']),fitted)
    return predict(features(image,cnn,xy),artifact['settled_regressor'])

def predict_surface_readout(image,lookup,geometry,xy,artifact):
    from scale_surface_geometry import envelope_features
    x,baseline,valid=envelope_features(image,lookup,geometry,xy)
    values=predict_readout(x,artifact['size_level_readout'])
    return torch.where(valid,values,torch.full_like(values,float('nan')))
