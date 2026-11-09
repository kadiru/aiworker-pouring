"""Source-verified calibration reel, clearly separate from autonomous task video."""
import argparse
import io
import json
import subprocess
from pathlib import Path, PurePosixPath
import h5py
import numpy as np
from PIL import Image, ImageDraw
from finish_media_source import verified_recording
from render_labeled_benchmark import text, sha

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/scale-A'
BG='#101820';WHITE='#F3F6F7';MUTED='#B3C0CC';GREEN='#67D6B1';RED='#FFAAA3'
FPS=20


def read(p):return json.loads(p.read_text())


def crop_box(spec,record):
    cal=record['cameras']['cup_detail']['calibration'];T=np.asarray(cal['T']);K=np.asarray(cal['K'])
    point=np.asarray(spec['cup_world'])+[0.,0.,spec['declared_cup_geometry']['rim_z_m']]
    v=(point-T[:3,3])@T[:3,:3]
    u,y=K[0,0]*v[0]/(-v[2])+K[0,2],K[1,2]-K[1,1]*v[1]/(-v[2])
    box=(round(u-208),round(y-180),round(u+208),round(y+236))
    assert min(box)>=0 and max(box)<=1024
    return box


def frame(ds,index,spec,record,number,total,box):
    im=Image.new('RGB',(1280,720),BG);d=ImageDraw.Draw(im)
    text(d,(28,15),'AIWORKER / CUP-SIZE CALIBRATION',17,MUTED,True)
    text(d,(28,44),f"{number:02}/{total:02}    Height {spec['cup_height_scale']:.2f}x   Diameter {spec['cup_diameter_scale']:.2f}x",24,WHITE,True)
    passed=record['label']['accepted'] and record['high_fill_coverage']
    text(d,(892,24),'CALIBRATION '+('PASS' if passed else 'FAIL'),22,GREEN if passed else RED,True,max_width=350)
    for camera,location,size in (('cup_detail',(28,84),608),('overview',(680,84),432)):
        picture=Image.open(io.BytesIO(ds['next_obs/'+camera+'_image'][index].tobytes())).convert('RGB')
        if camera=='cup_detail':picture=picture.crop(box)
        im.paste(picture.resize((size,size),Image.Resampling.LANCZOS),location)
    text(d,(680,540),'Authored filling, then settling',25,WHITE,True,max_width=570)
    text(d,(680,583),f"Final level: {record['label']['target']*100:.1f}% of rim",24,WHITE,max_width=570)
    text(d,(680,622),'Zero measured loss' if record['label']['accepted'] else 'Calibration checks failed',22,GREEN if passed else RED,max_width=570)
    text(d,(680,665),f"SIM {float(ds['next_time'][index]):05.2f}s   /   2x playback",19,MUTED,max_width=570)
    return im


def main():
    p=argparse.ArgumentParser();p.add_argument('--round',choices=('r1','r2'),required=True);a=p.parse_args()
    stem='pilot-'+a.round;audit=read(E/(stem+'-audit.json'));plan=read(E/('plan-'+a.round+'.json'))
    assert audit['complete'] and audit['plan_sha256']==sha(E/('plan-'+a.round+'.json'))
    specs={s['name']:s for s in plan['pilot']};assert set(specs)=={r['name'] for r in audit['records']}
    output=E/(stem+'.mp4');assert not output.exists()
    preview=E/(stem+'-video-review');preview.mkdir()
    proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
        '-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    count=0;clips=[];checks=[]
    def emit(im):
        nonlocal count
        proc.stdin.write(im.tobytes());count+=1
    card=Image.new('RGB',(1280,720),BG);d=ImageDraw.Draw(card)
    text(d,(60,90),'Preparing different cup sizes',40,WHITE,True)
    text(d,(60,190),'Native Isaac Sim calibration / all pilots included',25,MUTED)
    text(d,(60,278),'The robot is parked. A controlled jet fills each cup.',28,WHITE)
    text(d,(60,335),'These are calibration scenes, not autonomous pours.',28,WHITE)
    text(d,(60,430),'Purpose: stable, visible liquid levels for perception training.',26,MUTED)
    text(d,(60,535),'Unchanged liquid settings / Material A / original bottle',23,MUTED)
    try:
        for _ in range(3*FPS):emit(card)
        for number,record in enumerate(sorted(audit['records'],key=lambda r:r['name']),1):
            spec=specs[record['name']];box=crop_box(spec,record)
            identity=dict(record,relative_path=str(PurePosixPath(record['path']).relative_to('/workspace/outputs/fountain')))
            with verified_recording(identity) as path,h5py.File(path) as f:
                assert f.attrs['status']=='complete'
                ds=f['data/demo_0'];n=len(ds['next_time']);mapping=list(range(0,n,2))
                if mapping[-1]!=n-1:mapping.append(n-1)
                mapping.extend([n-1]*FPS);start=count
                selected={0,len(mapping)//3,2*len(mapping)//3,len(mapping)-1}
                for j,index in enumerate(mapping):
                    im=frame(ds,index,spec,record,number,len(specs),box)
                    if j in selected:
                        reference=preview/f'expected-{count:05}.png';im.save(reference)
                        checks.append(dict(frame=count,reference=str(reference),name=spec['name'],native_index=index))
                    emit(im)
                clips.append(dict(name=spec['name'],source_sha256=record['sha256'],start_frame=start,
                                  end_frame_exclusive=count,mapping=mapping,crop_box=box,playback_speed=2))
        proc.stdin.close();assert proc.wait()==0
    except BaseException:
        proc.kill();proc.wait();raise
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
    for c in checks:
        result=subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(output),'-vf',f"select=eq(n\\,{c['frame']})",
                               '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True)
        encoded=np.frombuffer(result.stdout,dtype=np.uint8).reshape(720,1280,3)
        expected=np.asarray(Image.open(c['reference']).convert('RGB'))
        c['mean_pixel_error']=float(np.abs(encoded.astype(float)-expected).mean())
        assert c['mean_pixel_error']<5.
        Image.fromarray(encoded).save(preview/f"encoded-{c['frame']:05}.png")
    report=dict(video=str(output),sha256=sha(output),bytes=output.stat().st_size,frames=count,fps=FPS,duration_s=count/FPS,
                clips=clips,comparison_checks=checks,full_decode_passed=True,manual_review='pending',
                audit_sha256=sha(E/(stem+'-audit.json')),helper_sha256=sha(Path(__file__)),
                scope='Complete native gradual-fill and settling calibration clips at labeled 2x, with final frame held one second. Offline seeds; not robot demonstrations or new policy evaluation. All declared pilots included.')
    (E/(stem+'-video.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('clips','comparison_checks')}))


if __name__=='__main__':main()
