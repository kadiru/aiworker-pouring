"""CPU-only validation error inspection for a preserved finish-study fit."""
import argparse,copy,json,shutil,sys
from pathlib import Path
import numpy as np,torch
sys.path[:0]=['/workspace/src','/workspace']
import pour_finish_train as trainer
from pour_finish_study import ROOT,PARENT
from pour_guard_models import CupEstimator
from pour_settled_model import features,predict
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
assert a.attempt.isalnum();source=ROOT/'fit-attempts'/a.attempt
out=ROOT/('diagnostic-'+a.attempt);out.mkdir(exist_ok=False)
shutil.copy2(source/'calibration-plan.json',out/'calibration-plan.json')
saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False);torch.set_num_threads(2)
anchors=json.loads((source/'calibration-anchor-manifest.json').read_text())['records']
mapping={}
for r in anchors:
    spec=json.loads((Path(r['path']).parent/'parameters.json').read_text());mapping[tuple(spec['cup_world'])+tuple(spec['source_xy'])]=r
trainer.ROOT=out;trainer.reference_crop_anchors=lambda saved,calibrations:mapping
data=trainer.build(saved)['validation'];model=CupEstimator().eval();model.load_state_dict(saved['cup_model'])
with torch.inference_mode():
    xs=[]
    for i in range(0,len(data['target']),64):
        im=data['image'][i:i+64];cnn=(model(im)+saved['cup_bias']).clamp(0,1)
        xs.append(features(im,cnn,data['cup_xy'][i:i+64]))
    x=torch.cat(xs);y=data['target'];recent=torch.tensor([n.startswith(('XW30_A_','FCAL_A_')) for n in data['episode']]);near=y>.8;mask=recent&near
    candidates=json.loads((source/'level-candidates.json').read_text());chosen=min(candidates,key=lambda c:c['score']);pred=predict(x,chosen['fit']);error=pred-y
    summaries=[]
    for delta in (0.,-.0025,-.005,-.0075,-.01):
        e=(pred+delta).clamp(0,1)-y
        summaries.append(dict(final_bias=delta,wide_near_min=float(e[mask].min()),wide_near_max=float(e[mask].max()),wide_near_abs_max=float(e[mask].abs().max()),mae=float(e.abs().mean()),near_mae=float(e[near].abs().mean())))
    worst=sorted(torch.where(mask)[0].tolist(),key=lambda i:-abs(float(error[i])))[:24];rows=[]
    sheet=Image.new('RGB',(1200,1000),'#101820');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(worst):
        row=dict(index=i,name=data['episode'][i],target=float(y[i]),prediction=float(pred[i]),error=float(error[i]));rows.append(row)
        xx=(j%6)*200;yy=(j//6)*250
        sheet.paste(Image.fromarray(data['image'][i].numpy()).resize((192,192)),(xx,yy+50))
        draw.text((xx+3,yy+2),row['name'].replace('XW30_A_','').replace('FCAL_A_',''),fill='white')
        draw.text((xx+3,yy+18),f"true {y[i]:.3f} / pred {pred[i]:.3f}",fill='white')
        draw.text((xx+3,yy+32),f"error {error[i]:+.3f}",fill='white')
    sheet.save(out/'worst-near-full.jpg',quality=95)
    assert not torch.cuda.is_initialized()
    result=dict(attempt=a.attempt,device='cpu',validation_frames=len(y),candidate=chosen['metrics'],post_output_calibration_probe=summaries,worst=rows)
    (out/'errors.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='worst'},indent=2))
