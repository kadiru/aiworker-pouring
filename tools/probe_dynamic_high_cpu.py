"""Read-only CPU diagnosis of retained models on reviewed high-fill training data."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from fit_scale_level import sha,read
from validate_pour_recording import h5py,decode
from scale_rim_refiner import locate_static
from pour_temporal_localization import consensus
from pour_guard_vision import plane_point,cup_crop
from scale_level_cnn import predict_size_cnn
from scale_color_mask import predict_mask
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
@torch.inference_mode()
def main():
    assert not torch.cuda.is_available();torch.set_num_threads(1)
    assert not (ROOT/'retained-high-cpu-probe.json').exists()
    protocol=read(ROOT/'protocol.json');specs={s['name']:s for s in protocol['seed']}
    loc=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False)
    primary=torch.load(ROOT/'primary-level.pt',map_location='cpu',weights_only=False)
    color=torch.load(ROOT/'color-mask-r2.pt',map_location='cpu',weights_only=False);rows=[]
    for part in ('seed_pilot','seed_remaining'):
        ap=ROOT/(part+'-r0-audit.json');review=read(ROOT/(part+'-r0-manual-review.json'));assert review['accepted'] and review['audit_sha256']==sha(ap)
        for r in read(ap)['records']:
            s=specs[r['name']];path=Path(r['path']);assert sha(path)==r['sha256']
            row=dict(name=r['name'],target=r['label']['target'])
            with h5py.File(path) as f:
                d=f['data/demo_0'];history=[];accepted=None
                for i in range(12):
                    rgb=decode(d['obs/cup_detail_image'][i],'jpeg95');uv,score=locate_static(rgb,loc['bank'],device='cpu')
                    history.append((uv.tolist(),score));accepted=consensus(history,loc['threshold'],loc['consensus'])
                    if accepted is not None:break
                row['confirmed']=accepted is not None
                if accepted is not None:
                    xy=plane_point(accepted['uv'],1.04+s['declared_cup_geometry']['rim_z_m'],loc['calibration'])[:2]
                    row['localization_error_mm']=float(np.linalg.norm(xy-s['cup_world'][:2])*1000)
                    cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['next_obs/cup_detail_world_T_camera'][-1].tolist(),table_z=1.04,cup_geometry=s['declared_cup_geometry'])
                    ids=np.flatnonzero(d['next_time'][:]>=d['next_time'][-1]-1.-1e-7)[::2]
                    x=torch.tensor(np.stack([cup_crop(decode(d['next_obs/cup_detail_image'][int(i)],'jpeg95'),xy,cal) for i in ids]));xy=torch.tensor(xy,dtype=torch.float32)[None].expand(len(x),2)
                    rgb=predict_size_cnn(x,xy,s['declared_cup_geometry'],primary);mask,valid=predict_mask(x,xy,s['declared_cup_geometry'],color)
                    row.update(primary_max_error=float((rgb-r['label']['target']).abs().max()),color_max_error=float((mask-r['label']['target']).abs().max()),color_valid=bool(valid.all()),max_disagreement=float((rgb-mask).abs().max()))
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(cases=rows,scope='CPU diagnosis of retained models on high-fill training episodes only. Does not accept models or replace the requested 4070 validation.',helper_sha256=sha(Path(__file__)),primary_sha256=sha(ROOT/'primary-level.pt'),color_sha256=sha(ROOT/'color-mask-r2.pt'),locator_sha256=sha(ROOT/'cup-neural-localization.pt'))
    (ROOT/'retained-high-cpu-probe.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
