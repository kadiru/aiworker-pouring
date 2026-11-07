"""Eight native views per development attempt, with honest final-score labels."""
import argparse,hashlib,io,json
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw
from render_labeled_benchmark import font

PROJECT=Path(__file__).resolve().parents[1]
DATA=PROJECT/'data/material-A-finish'
from finish_media_source import verified_recording


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--progress',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();progress=read(a.progress);integrity=read(a.manifest)
    assert progress['part']=='development' and progress['stage']=='complete' and len(progress['rows'])==8
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    indexed={r['name']:r for r in integrity['files']};a.output.mkdir(exist_ok=False)
    cases=[];checkpoints=set()
    for row in sorted(progress['rows'],key=lambda r:r['name']):
        identity=indexed[row['name']];folder=(DATA/identity['relative_path']).parent
        assert identity['valid']
        decisions=read(folder/'policy-decisions.json');sheet=Image.new('RGB',(1536,960),'#101820');draw=ImageDraw.Draw(sheet)
        with verified_recording(identity) as path,h5py.File(path) as f:
            spec=json.loads(f.attrs['spec']);metrics=json.loads(f.attrs['metrics'])
            assert f.attrs['status']=='complete' and spec['name']==row['name'] and spec['protocol']==18
            checkpoint=metrics['policy']['checkpoint_sha256'];checkpoints.add(checkpoint)
            assert checkpoint==row['checkpoint_sha256']
            d=f['data/demo_0'];times=d['next_time'][:];n=len(times);assert len(decisions)==n
            verified=np.flatnonzero(d['ground_truth/grasp_verified'][:]);stops=np.flatnonzero(d['ground_truth/stop_command'][:])
            grip=int(verified[0]) if len(verified) else int(np.argmax(d['ground_truth/bottle_pose'][:,2]))
            stop=int(stops[0]) if len(stops) else n-1
            returned=int(np.argmin(abs(times-min(times[-1],times[stop]+1.3))))
            release=next((i for i,r in enumerate(decisions) if r['mode']=='Releasing bottle'),n-1)
            views=[('obs',0,'task','Initial'),('next_obs',grip,'wrist','Grip' if len(verified) else 'No verified grip'),
                ('next_obs',(grip+stop)//2,'overview','Between grip and stop'),('next_obs',stop,'task','Stop / task'),
                ('next_obs',stop,'wrist','Stop / wrist'),('next_obs',returned,'overview','After stop'),
                ('next_obs',release,'task','Release' if release<n-1 else 'Final fallback'),('next_obs',n-1,'task','Final')]
            status='PASS' if row['strict_pass'] else 'FAIL';color='#67D6B1' if row['strict_pass'] else '#FFAAA3'
            draw.text((12,8),f"{row['name']}  |  DEVELOPMENT  |  {status}",font=font(23,True),fill=color)
            result=f"{row['stop_reason']} | rim {100*row['cup_surface_height_fraction']:.2f}% | source {row['source_remaining_ml']:.4f} | final outside {row['final_outside_ml']:.6f} nominal mL"
            draw.text((12,42),result,font=font(18),fill='white')
            draw.text((12,70),', '.join(row['failure_reasons']) or 'All strict task checks passed',font=font(17),fill=color)
            samples=[]
            for j,(group,i,camera,label) in enumerate(views):
                x=(j%4)*384;y=112+(j//4)*424
                image=Image.open(io.BytesIO(d[group+'/'+camera+'_image'][i].tobytes())).convert('RGB');assert image.size==(384,384)
                time=float(d['time' if group=='obs' else 'next_time'][i])
                draw.text((x+8,y),f'{label} / {time:.2f}s',font=font(16),fill='#B3C0CC');sheet.paste(image,(x,y+30))
                samples.append(dict(group=group,index=i,camera=camera,t=time,label=label))
        output=a.output/(row['name']+'.jpg');sheet.save(output,quality=95)
        cases.append(dict(name=row['name'],source_sha256=identity['sha256'],sheet=str(output),sheet_sha256=sha(output),
            strict_pass=row['strict_pass'],failure_reasons=row['failure_reasons'],samples=samples,manual_review='pending'))
    assert len(checkpoints)==1
    report=dict(checkpoint_sha256=next(iter(checkpoints)),cases=cases,manual_review='pending',
        source_manifest_sha256=sha(a.manifest),progress_sha256=sha(a.progress),
        scope='Eight selected native frames per development attempt; not full-video playback. No HDF5 attributes changed.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),selected_frames=sum(len(r['samples']) for r in cases),manual_review='pending')))


if __name__=='__main__':main()
