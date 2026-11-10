"""Freeze reviewed tracker crops for repeated development fits; no reserved images."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from dynamic_level_data import ROOT,collect,read,sha
from dynamic_tracker_data import entries
from validate_pour_recording import h5py,decode
from pour_guard_vision import cup_crop
from scale_level_cnn import predict_size_cnn
from scale_color_mask import predict_mask
from fit_dynamic_rim import protected_unchanged

def main():
    assert not (ROOT/'level-cache-v1.json').exists() and not (ROOT/'component-selection.json').exists()
    cfg=read(ROOT/'pair-r4-protocol.json');assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    protected_unchanged();torch.set_num_threads(4);assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();started=time.perf_counter()
    originals={n:(ROOT/n).read_bytes() for n in ('level-localization-check.json','dynamic-localization-check.json')}
    try:
        protocol,parent,data=collect(cfg['parts'],cfg['locator'],cfg['locator_selection'])
        for n in originals:(ROOT/('cache-'+n)).write_bytes((ROOT/n).read_bytes())
    finally:
        for n,content in originals.items():(ROOT/n).write_bytes(content)
    arrays={};metadata={}
    for split,a in data.items():
        for name in ('x','xy','y'):arrays[split+'_'+name]=a[name].cpu().numpy()
        metadata[split]=dict(meta=a['meta'],geometry=a['geometry'],target=a['target'])
    np.savez_compressed(ROOT/'level-cache-v1.npz',**arrays)
    (ROOT/'level-cache-v1-meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Counterfactual exact-position crops are diagnostic only; they never enter the cache.
    rgb=torch.load(ROOT/'pair-r4-rgb.pt',map_location='cpu',weights_only=False);color=torch.load(ROOT/'pair-r4-color.pt',map_location='cpu',weights_only=False)
    indexed={s['name']:(s,r) for s,r,split in entries()};diagnostic=[];panels=[]
    targets=['SD_A_motion_validation_r0_05_low','SD_A_motion_validation_r0_01_low','SC_A_static_validation_03_01','SC_A_static_validation_05_01']
    for name in targets:
        a=data['validation'];ids=[i for i,m in enumerate(a['meta']) if m['name']==name];s,r=indexed[name]
        with h5py.File(r['path']) as f:
            d=f['data/demo_0'];cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,cup_geometry=s['declared_cup_geometry'])
            for i in (ids[0],ids[-1]):
                image=decode(d['next_obs/cup_detail_image'][a['meta'][i]['index']],'jpeg95');exact=cup_crop(image,s['cup_world'][:2],cal)
                native=a['x'][i];batch=torch.stack((native,torch.tensor(exact,device='cuda')));xy=torch.tensor([a['xy'][i].tolist(),s['cup_world'][:2]],device='cuda')
                rp=predict_size_cnn(batch,xy,s['declared_cup_geometry'],rgb).cpu().tolist();cp,valid=predict_mask(batch,xy,s['declared_cup_geometry'],color)
                row=dict(name=name,index=a['meta'][i]['index'],target=a['target'][i],tracked_xy=a['xy'][i].cpu().tolist(),exact_xy=s['cup_world'][:2],rgb_predictions=rp,color_predictions=cp.cpu().tolist(),color_valid=valid.cpu().tolist());diagnostic.append(row)
                panels.append((row,native.cpu().numpy(),exact))
    canvas=Image.new('RGB',(900,len(panels)*124),'white');draw=ImageDraw.Draw(canvas)
    for j,(row,predicted,exact) in enumerate(panels):
        y=j*124;draw.text((5,y+2),row['name']+' frame '+str(row['index'])+' target '+format(row['target'],'.3f'),fill='black')
        for k,im in enumerate((predicted,exact)):
            x=k*450;canvas.paste(Image.fromarray(im.astype(np.uint8)),(x+5,y+24));draw.text((x+110,y+28),('TRACKER CROP' if k==0 else 'EXACT-POSE DIAGNOSTIC ONLY')+'\nRGB %.3f; color %.3f'%(row['rgb_predictions'][k],row['color_predictions'][k]),fill='black')
    canvas.save(ROOT/'level-cache-crop-diagnostic.png')
    (ROOT/'level-cache-crop-diagnostic.json').write_text(json.dumps(dict(rows=diagnostic,preview_sha256=sha(ROOT/'level-cache-crop-diagnostic.png'),scope='Read-only counterfactual pose diagnostic; exact-position crops are excluded from training/cache and cannot grant acceptance.'),indent=2)+'\n')
    manifest=dict(passed=True,tracker_sha256=sha(ROOT/cfg['locator']),tracker_module_sha256=sha(ROOT/'dynamic_cup_tracker.py'),source_protocol_sha256=sha(ROOT/'pair-r4-protocol.json'),cache_sha256=sha(ROOT/'level-cache-v1.npz'),metadata_sha256=sha(ROOT/'level-cache-v1-meta.json'),admission={n:sha(ROOT/('cache-'+n)) for n in originals},counts={s:dict(episodes=len({m['name'] for m in a['meta']}),frames=len(a['y'])) for s,a in data.items()},helper_sha256=sha(Path(__file__)),total_seconds=time.perf_counter()-started,scope='Only tracker-derived RGB crops, known geometry and image-derived positions enter training inputs; measured levels are labels. Whole-episode train/validation splits unchanged. No reserved images.')
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items());protected_unchanged()
    (ROOT/'level-cache-v1.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
