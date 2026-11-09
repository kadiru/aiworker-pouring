"""Export audited native task keyframes; manual inspection remains pending."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);a=p.parse_args()
    outcomes=read(ROOT/(a.part+'-outcomes.json'));assert outcomes['status']=='complete'
    candidate=read(ROOT/'candidate.json');assert outcomes['checkpoint_sha256']==candidate['checkpoint_sha256']
    audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete']
    indexed={r['name']:r for r in audit['records']};out=ROOT/(a.part+'-review');out.mkdir(exist_ok=False);records=[]
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default()
    for result in outcomes['cases']:
        name=result['name'];r=indexed[name];path=Path(r['path']);assert sha(path)==r['sha256']==result['source_sha256']
        samples=read(path.parent/'samples.json');decisions=read(path.parent/'policy-decisions.json')
        times=np.array([s['t'] for s in samples]);assert len(samples)==len(decisions)
        first=lambda pred:next((i for i,d in enumerate(decisions) if pred(d)),None)
        grasp=first(lambda d:d.get('visual_grasp_confirmed',False));stop=first(lambda d:d.get('stop_latched') is not None)
        event=next((i for i,s in enumerate(samples) if s['ever_table_ml'] or s['spilled_ml']),None)
        maximum=int(np.argmax([abs(s['bottle_tilt_deg']) for s in samples]))
        selected=[('Initial',0,'overview'),('Grasp' if grasp is not None else 'No confirmed grasp',grasp or 0,'wrist'),
            ('Maximum tilt',maximum,'overview'),('First outside/table' if event is not None else 'Maximum tilt detail',event if event is not None else maximum,'cup_detail'),
            ('Stop' if stop is not None else 'No stop',stop if stop is not None else len(samples)-1,'overview'),
            ('Stop liquid level' if stop is not None else 'Final liquid level',stop if stop is not None else len(samples)-1,'cup_detail')]
        for mode,camera in (('Returning upright','overview'),('Setting bottle down','overview'),('Releasing bottle','wrist'),('Retreating','overview')):
            index=first(lambda d:d['mode']==mode and d.get('stop_latched') is not None)
            selected.append((mode if index is not None else 'Missing: '+mode,index if index is not None else len(samples)-1,camera))
        selected.extend([('Final pose',len(samples)-1,'overview'),('Final liquid level',len(samples)-1,'cup_detail')])
        canvas=Image.new('RGB',(1152,1730),(246,246,246));draw=ImageDraw.Draw(canvas)
        draw.text((10,8),name+' | '+a.part.upper()+' '+('PASS' if result['accepted'] else 'FAIL'),font=font,fill=(20,100,45) if result['accepted'] else (160,40,30))
        reasons=' / '.join(result['acceptance_failures']) or 'All strict task checks passed'
        draw.text((10,29),reasons,font=font,fill=(30,30,30));frames=[]
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            for j,(label,index,camera) in enumerate(selected):
                rgb=decode(d['next_obs/'+camera+'_image'][index],'jpeg95');x=(j%3)*384;y=54+(j//3)*418
                draw.text((x+6,y+5),f'{label} | {times[index]:.2f}s',font=font,fill=(30,30,30))
                image=Image.fromarray(rgb).resize((384,384),Image.Resampling.BILINEAR)
                canvas.paste(image,(x,y+28));frames.append(dict(label=label,index=index,camera=camera,t=float(times[index])))
        strip=None;return_frames=[]
        if stop is not None:
            duration=next((d['side_return_plan']['upright_seconds'] for d in decisions if d.get('side_return_plan')),None)
            if duration is not None:
                strip=Image.new('RGB',(1536,1254),(246,246,246));sd=ImageDraw.Draw(strip)
                with h5py.File(path) as f:
                    images=f['data/demo_0/next_obs/overview_image']
                    for j,dt in enumerate(np.linspace(0,duration,12)):
                        index=int(np.argmin(abs(times-(times[stop]+dt))))
                        x=(j%4)*384;y=(j//4)*418
                        sd.text((x+6,y+5),f"Return +{dt:.2f}s | tilt {samples[index]['bottle_tilt_deg']:.1f} deg",font=font,fill=(30,30,30))
                        strip.paste(Image.fromarray(decode(images[index],'jpeg95')).resize((384,384)),(x,y+28))
                        return_frames.append(dict(index=index,camera='overview',t=float(times[index])))
                strip.save(out/(name+'-return.jpg'),quality=94)
        dest=out/(name+'.jpg');canvas.save(dest,quality=94)
        records.append(dict(name=name,source_sha256=r['sha256'],strict_pass=result['strict_pass'],accepted=result['accepted'],failure_reasons=result['failure_reasons'],acceptance_failures=result['acceptance_failures'],
            frames=frames,return_frames=return_frames,reviewed_files={str(dest):sha(dest),**({str(out/(name+'-return.jpg')):sha(out/(name+'-return.jpg'))} if strip is not None else {})},manual_review='pending'))
    report=dict(checkpoint_sha256=candidate['checkpoint_sha256'],cases=records,manual_review='pending',audit_sha256=sha(ROOT/'recording-audit.json'),
        helper_sha256=sha(Path(__file__)),scope='Twelve task keyframes and, when available, twelve evenly spaced return frames per attempt, resized only for display. Not continuous playback. Independent raw-sample scoring retains every failure. Manual review must be recorded only after inspection.')
    (ROOT/(a.part+'-visual-review.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),selected_native_frames=sum(len(r['frames'])+len(r['return_frames']) for r in records),manual_review='pending')))
if __name__=='__main__':main()
