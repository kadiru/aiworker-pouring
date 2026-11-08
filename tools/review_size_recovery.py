"""Export labeled native keyframes, without claiming continuous video review."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from PIL import Image,ImageDraw
from validate_pour_recording import h5py,decode
from pour_size_study import ROOT,OUTPUTS
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons


def main():
    out=ROOT/'recovery-review';out.mkdir(exist_ok=True);records=[]
    for path in sorted(OUTPUTS.glob('*/*SZ_A_recovery_*/episode.hdf5')):
        folder=path.parent
        read=lambda n:json.loads((folder/n).read_text())
        spec=read('parameters.json');samples=read('samples.json');decisions=read('policy-decisions.json');metrics=read('metrics.json')
        r=completed_row(folder,metrics['policy']['checkpoint_sha256']);passed=strict_success(r)
        times=np.array([x['t'] for x in samples])
        first=lambda test:next((i for i,x in enumerate(decisions) if test(x)),None)
        stop_found=first(lambda x:x.get('stop_latched') is not None)
        stop=len(samples)-1 if stop_found is None else stop_found
        grasp=first(lambda x:x.get('visual_grasp_confirmed',False))
        place=first(lambda x:x['mode']=='Setting bottle down');release=first(lambda x:x['mode']=='Releasing bottle')
        loss=next((i for i,x in enumerate(samples) if x['ever_table_ml']),None)
        event=loss if loss is not None else int(np.argmax([abs(x['bottle_tilt_deg']) for x in samples]))
        selected=[('Initial',0,'task'),('No grasp confirmation' if grasp is None else 'Grasp confirmed',0 if grasp is None else grasp,'wrist'),
            ('Before loss' if loss is not None else 'Maximum tilt',max(0,event-1) if loss is not None else event,'task'),
            ('First worktop loss' if loss is not None else 'Wrist view',event,'wrist'),
            ('No stop decision' if stop_found is None else 'Stop decision',stop,'task'),
            ('After stop' if stop_found is not None else 'Final control state',int(np.argmin(abs(times-(times[stop]+1.1)))),'task'),
            ('No placement' if place is None else 'Place',len(samples)-1 if place is None else place,'task'),
            ('No release' if release is None else 'Release',len(samples)-1 if release is None else release,'wrist'),
            ('Final',len(samples)-1,'overview')]
        sheet=Image.new('RGB',(1152,1322),(246,246,246));draw=ImageDraw.Draw(sheet)
        draw.text((10,8),spec['name']+' | DEVELOPMENT '+('PASS' if passed else 'FAIL'),fill=(20,100,45) if passed else (150,35,25))
        draw.text((10,25),' / '.join(reasons(r)) if not passed else 'Full task checks passed in this consumed layout',fill=(30,30,30))
        frames=[]
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            for j,(label,i,camera) in enumerate(selected):
                rgb=decode(d['next_obs/'+camera+'_image'][i],'jpeg95')
                x=(j%3)*384;y=50+(j//3)*424
                draw.text((x+6,y+7),f'{label} | {times[i]:.2f}s | {camera}',fill=(30,30,30))
                sheet.paste(Image.fromarray(rgb),(x,y+40))
                frames.append(dict(label=label,index=i,camera=camera,t=float(times[i])))
        dest=out/(spec['name']+'.jpg');sheet.save(dest,quality=92)
        records.append(dict(name=spec['name'],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sheet_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),strict_pass=passed,
            failure_reasons=reasons(r),frames=frames,manual_review='pending'))
    (ROOT/'recovery-keyframes.json').write_text(json.dumps(dict(scope='Selected native frames only, not continuous playback or synthetic camera imagery.',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=records),indent=2)+'\n')
    print(json.dumps(dict(closed_cases=len(records),keyframes=sum(len(r['frames']) for r in records))))


if __name__=='__main__':main()
