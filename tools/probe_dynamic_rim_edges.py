"""Fixed geometric rim-edge diagnostic; declared size, RGB location, no pose input."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from fit_dynamic_rim import ROOT,read,sha,collect,protected_unchanged,CupLocator
from pour_guard_vision import plane_point

def outline(center,cal,height,radius,origin):
    # Match the authored 20-sided visual shell, sampling each straight edge.
    xy=plane_point(center,cal['table_z']+height,cal)[:2]
    theta=np.arange(20)*2*np.pi/20
    vertices=np.stack((np.cos(theta),np.sin(theta)),1)*(radius+.0024)
    fraction=np.arange(8)/8
    ring=(vertices[:,None]*(1-fraction)[None,:,None]+np.roll(vertices,-1,axis=0)[:,None]*fraction[None,:,None]).reshape(-1,2)
    world=np.c_[ring+xy,np.full(len(ring),cal['table_z']+height)]
    t=np.asarray(cal['T']);k=np.asarray(cal['K']);v=(world-t[:3,3])@t[:3,:3]
    uv=np.stack((k[0,0]*v[:,0]/-v[:,2]+k[0,2],k[1,2]-k[1,1]*v[:,1]/-v[:,2]),1)-origin
    tangent=np.roll(uv,-1,axis=0)-np.roll(uv,1,axis=0)
    normal=np.stack((-tangent[:,1],tangent[:,0]),1);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    return uv,normal

def refine(image,center,origin,cal,height,radius):
    device=image.device
    gray=(image.float()*image.new_tensor([.299,.587,.114],dtype=torch.float32)).sum(-1)[None,None]/255
    gray=F.avg_pool2d(gray,3,1,1)
    grad=torch.cat((F.pad((gray[:,:,:,2:]-gray[:,:,:,:-2])/2,(1,1,0,0)),F.pad((gray[:,:,2:,:]-gray[:,:,:-2,:])/2,(0,0,1,1))),1)
    point=np.asarray(center).copy();last_score=None
    for extent,step in ((24.,2.),(2.,.5)):
        uv,normal=outline(point,cal,height,radius,origin)
        axis=torch.arange(-extent,extent+step/2,step,device=device)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij');shift=torch.stack((xx.flatten(),yy.flatten()),1)
        grid=torch.tensor(uv,device=device,dtype=torch.float32)[None]+shift[:,None]
        values=F.grid_sample(grad,(grid/255*2-1)[None],align_corners=True)[0].permute(1,2,0)
        normals=torch.tensor(normal,device=device,dtype=torch.float32)
        response=(values*normals[None]).sum(-1).abs().clamp(max=.12)
        # Trim both tails so a few occluded/high-contrast bottle edges cannot dominate.
        response=response.sort(1).values[:,16:-16].mean(1)
        selected=int(response.argmax());point+=shift[selected].cpu().numpy();last_score=float(response[selected])
    return point,last_score

def main():
    assert not (ROOT/'rim-edge-probe.json').exists();protected_unchanged();torch.set_num_threads(4)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    model=CupLocator().cuda().eval();model.load_state_dict(torch.load(ROOT/'coarse-d2.pt',map_location='cpu',weights_only=False)['state'])
    data,cal=collect(model,manifest_name='rim-edge-data.json');report={}
    with torch.inference_mode():
        for split,a in data.items():
            rows=[]
            for i,(im,m,c,o) in enumerate(zip(a['x'],a['meta'],a['coarse'],a['origin'])):
                # Only declared dimensions/calibration and image-predicted center enter refinement.
                uv,score=refine(im,c,np.asarray(o),cal,.004+.034*m['cup_height_scale'],.024*m['cup_diameter_scale'])
                error=float(np.linalg.norm(plane_point(uv,m['z'],cal)[:2]-m['xy'])*1000)
                rows.append(dict(name=m['name'],group=m['group'],index=m['index'],uv=uv.tolist(),edge_score=score,error_mm=error))
                if i%1000==0:print(json.dumps(dict(split=split,frames=i+1)),flush=True)
            report[split]=dict(max_error_mm=max(r['error_mm'] for r in rows),mean_error_mm=float(np.mean([r['error_mm'] for r in rows])),failures=sum(r['error_mm']>=3 for r in rows),frames=len(rows),rows=rows)
    report.update(optimizer_steps=0,autonomy_accepted=False,scope='Fixed geometric edge-alignment development diagnostic, not accepted localization. No validation fitting, ground-truth inference crops, exclusions or reserved images.',helper_sha256=sha(ROOT/'probe_dynamic_rim_edges.py'),coarse_sha256=sha(ROOT/'coarse-d2.pt'))
    protected_unchanged();(ROOT/'rim-edge-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in report[s].items() if k!='rows'} for s in ('train','validation')}),flush=True)
if __name__=='__main__':main()
