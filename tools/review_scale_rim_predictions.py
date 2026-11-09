"""Render all 18 final validation predictions for separate manual review."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import project
from scale_rim_refiner import crop_native
ROOT=Path('/workspace/learning/A-scale-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=read(ROOT/'cup-neural-localization.json');assert report['passed']
    m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');audit=read(ROOT/m['collection_audit'])
    assert report['plan_sha256']==sha(ROOT/'plan.json') and report['audit_sha256']==sha(ROOT/m['collection_audit'])
    records={r['name']:r for r in audit['records']};specs={s['name']:s for s in plan['collect']}
    rows=[r for r in report['selected']['rows'] if r['group']=='next_obs' and r['index']==-1]
    assert {r['name'] for r in rows}=={s['name'] for s in plan['collect'] if s['study_split']=='validation'}
    output=ROOT/'rim-r6-review';output.mkdir();panels=[];entries=[];sheets=[]
    for r in rows:
        source=records[r['name']];spec=specs[r['name']];assert sha(Path(source['path']))==source['sha256']
        with h5py.File(source['path']) as f:
            d=f['data/demo_0'];image=decode(d['next_obs/cup_detail_image'][-1],'jpeg95')
            cal=dict(K=f['calibration/cup_detail/intrinsics'][:],T=d['next_obs/cup_detail_world_T_camera'][-1])
        truth=project(np.r_[spec['cup_world'][:2],1.04+spec['declared_cup_geometry']['rim_z_m']],cal)
        crop,origin=crop_native(image,r['uv']);p=Image.fromarray(crop);draw=ImageDraw.Draw(p)
        for uv,color in ((truth,'lime'),(r['uv'],'cyan')):
            x,y=np.asarray(uv)-origin;draw.line((x-5,y,x+5,y),fill=color,width=1);draw.line((x,y-5,x,y+5),fill=color,width=1)
        panels.append(p);entries.append(dict(name=r['name'],error_mm=r['error_mm'],recording_sha256=source['sha256'],group='next_obs',index=-1))
    for start in range(0,len(rows),6):
        canvas=Image.new('RGB',(960,620),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),'Validation: green projected truth; cyan refined RGB estimate | native pixel scale',fill='black')
        for j,p in enumerate(panels[start:start+6]):
            x=(j%3)*320;y=(j//3)*294+24;canvas.paste(p,(x+32,y+32));r=rows[start+j]
            draw.text((x+4,y),r['name'],fill='black');draw.text((x+4,y+15),f'{r["error_mm"]:.3f} mm | static known-size cup',fill='black')
        path=output/f'sheet-{start//6:02d}.png';canvas.save(path);sheets.append(dict(path=str(path),sha256=sha(path)))
    result=dict(manual_review='pending',cases=entries,sheets=sheets,fit_report_sha256=sha(ROOT/'cup-neural-localization.json'),
        helper_sha256=sha(Path(__file__)),scope='One actual final validation camera frame per episode, native-pixel model-centered crop; stored evaluated predictions, not autonomous demonstrations or reserved tests.')
    (ROOT/'rim-r6-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(frames=len(rows),sheets=len(sheets))))
if __name__=='__main__':main()
