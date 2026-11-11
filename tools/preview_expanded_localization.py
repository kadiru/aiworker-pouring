"""Native RGB diagnostic only: labeled poses never enter inference inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from fit_dynamic_rim import ROOT,read,sha
from perception_repair_data import entries
from validate_pour_recording import h5py,decode
from pour_guard_vision import project
from scale_rim_refiner import crop_native

def main():
    report=read(ROOT/'tracker-t2.json');previous=read(ROOT/'static-repair-tracker-admission.json')
    failed=[r['name'] for r in previous['cases'] if not r['passed']]
    validation=sorted((r for r in report['cases'] if r['split']=='validation'),key=lambda r:r['tracked_max_mm'] or 100,reverse=True)
    selected=failed+[r['name'] for r in validation[:2]];assert len(selected)==8
    source={s['name']:(s,r) for s,r,split in entries()};canvas=Image.new('RGB',(1024,616),'white');draw=ImageDraw.Draw(canvas);evidence=[]
    calibration=torch.load(ROOT/'tracker-t2.pt',map_location='cpu',weights_only=False)['calibration']
    for j,name in enumerate(selected):
        spec,record=source[name];row=next(r for r in report['rows'] if r['name']==name and r['valid'])
        uv=np.array(row['uv'])
        truth=project(np.r_[spec['cup_world'][:2],calibration['table_z']+spec['declared_cup_geometry']['rim_z_m']],calibration)
        with h5py.File(record['path']) as f:image=decode(f['data/demo_0/'+row['group']+'/cup_detail_image'][row['index']],'jpeg95')
        crop,origin=crop_native(image,uv);panel=Image.fromarray(crop);pd=ImageDraw.Draw(panel)
        for point,color in ((truth,'lime'),(uv,'magenta')):
            x,y=point-origin;pd.line((x-6,y,x+6,y),fill=color,width=2);pd.line((x,y-6,x,y+6),fill=color,width=2)
        x=(j%4)*256;y=(j//4)*308;canvas.paste(panel,(x,y+52))
        draw.text((x+3,y+3),name.replace('SD_A_motion_','')+'\nerror %.3f mm'%row['error_mm'],fill='black')
        evidence.append(dict(name=name,error_mm=row['error_mm'],source_sha256=record['sha256'],group=row['group'],index=row['index']))
    canvas.save(ROOT/'expanded-localization-review.png')
    (ROOT/'expanded-localization-preview.json').write_text(json.dumps(dict(rows=evidence,legend='green=projected reference; magenta=predicted rim center; exact 256-pixel native crops',preview_sha256=sha(ROOT/'expanded-localization-review.png')),indent=2)+'\n')

if __name__=='__main__':main()
