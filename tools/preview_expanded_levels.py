"""Show exact cached inputs for the worst r8 validation episodes."""
import json
import numpy as np
from PIL import Image,ImageDraw
from fit_dynamic_rim import ROOT,read,sha

def main():
    reports={k:read(ROOT/('pair-r8-'+k+'.json')) for k in ('rgb','color')}
    chosen=[]
    for kind,r in reports.items():
        for n,e in sorted(r['selected']['per_episode_mae']['validation'].items(),key=lambda v:-v[1])[:4]:
            if n not in chosen:chosen.append(n)
    meta=read(ROOT/'level-cache-v4-meta.json')['validation'];a=np.load(ROOT/'level-cache-v4.npz')
    x=a['validation_x'];y=a['validation_y'];rows=[]
    canvas=Image.new('RGB',(320*len(chosen),352),'white');draw=ImageDraw.Draw(canvas)
    for j,name in enumerate(chosen):
        indices=[i for i,m in enumerate(meta['meta']) if m['name']==name];i=indices[-1];m=meta['meta'][i]
        preds={k:next(v['prediction'] for v in r['rows'] if v['name']==name and v['index']==m['index']) for k,r in reports.items()}
        canvas.paste(Image.fromarray(x[i]).resize((288,288),Image.Resampling.NEAREST),(j*320,64))
        label=name.replace('SD_A_motion_','').replace('SC_A_static_','')
        draw.text((j*320+2,3),label+'\nmeasured %.2f%%\nRGB %.2f%%; color %.2f%%'%(y[i]*100,preds['rgb']*100,preds['color']*100),fill='black')
        rows.append(dict(name=name,index=m['index'],target=float(y[i]),predictions=preds))
    canvas.save(ROOT/'expanded-level-r8-review.png')
    (ROOT/'expanded-level-r8-preview.json').write_text(json.dumps(dict(rows=rows,preview_sha256=sha(ROOT/'expanded-level-r8-review.png'),cache_sha256=sha(ROOT/'level-cache-v4.npz'),scope='Exact model inputs enlarged with nearest-neighbor; development diagnosis only.'),indent=2)+'\n')

if __name__=='__main__':main()
