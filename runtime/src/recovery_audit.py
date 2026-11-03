"""CPU checks and actual-camera review sheets for automatic expert recoveries."""
import json
from pathlib import Path
import sys
sys.path.append('/workspace/pydeps51')
import h5py
import numpy as np
from PIL import Image,ImageDraw
from validate_pour_recording import validate,decode
from pour_recovery_study import ROOT


def audit():
    latest={}
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*R_A_*/episode.hdf5')):
        params=json.loads((path.parent/'parameters.json').read_text())
        if params.get('expert_recovery'):latest[params['name']]=path
    reports=[]
    for name,path in sorted(latest.items()):
        report=validate(path)
        (path.parent/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(dict(name=name,**report))
        with h5py.File(path,'r') as f:
            d=f['data/demo_0'];n=len(d['actions']);mask=d['expert_supervision'][:]
            expert=np.flatnonzero(mask);stops=np.flatnonzero(d['ground_truth/stop_command'][:])
            indices=[0,int(np.argmin(abs(d['next_time'][:]-3.))),int(expert[0]) if len(expert) else n-1]
            flow=np.r_[0.,-np.diff(d['ground_truth/bottle_ml'][:])];flow[~mask.astype(bool)]=0.
            indices.extend([int(np.argmax(flow)),int(stops[0]) if len(stops) else n-1,n-1])
            for camera in ('task','wrist'):
                image=Image.new('RGB',(1536,280),(24,30,34));draw=ImageDraw.Draw(image)
                for column,(label,index) in enumerate(zip(('initial','grasp','takeover','peak flow','stop','final'),indices)):
                    rgb=decode(d['next_obs/'+camera+'_image'][index],f.attrs.get('image_encoding','raw'))
                    image.paste(Image.fromarray(rgb).resize((256,256)),(column*256,24))
                    draw.text((column*256+4,5),label+f' {float(d["next_time"][index]):.2f}s',fill='white')
                image.save(ROOT/f'review-{name}-{camera}.jpg',quality=94)
    (ROOT/'collection-audit.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],physical=r['physical_success'],
        recovery_zero_spill=r['metrics'].get('recovery_success_zero_spill'),
        admitted=r.get('recovery_training_eligible'),gates=r['metrics']['gates']) for r in reports],indent=2))


if __name__=='__main__':audit()
