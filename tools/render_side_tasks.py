"""Complete, honestly labeled native task reels; local CPU composition only."""
import argparse,json,subprocess
from pathlib import Path,PurePosixPath
import h5py
from PIL import Image,ImageDraw
from render_labeled_benchmark import text,sha
from render_finish_videos import picture,phase_label,verdict
from render_layout_variety import lines
from finish_media_source import verified_recording
from pour_wide_report import strict_success
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/side-return-A'
FPS=20;SIZE=(1280,720);BG='#101820';WHITE='#F3F6F7';MUTED='#B3C0CC';GREEN='#67D6B1';RED='#FFAAA3'
def read(p):return json.loads(p.read_text())
def card(report,part):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im);c=report['counts'];n=c['completed']
    text(d,(56,40),'AIWORKER / ISAAC SIM',18,MUTED,True)
    text(d,(56,90),'Grasp. Pour. Stop. Put the bottle down.',36,WHITE,True)
    text(d,(56,156),part.upper()+' / ALL ATTEMPTS INCLUDED',18,MUTED,True)
    text(d,(56,224),f"{c['accepted']} / {n}",72,WHITE,True)
    text(d,(56,310),'complete tasks passed',23,MUTED)
    text(d,(56,372),f"{c['zero_measured_loss']} / {n} zero final / table loss",25,GREEN,True)
    text(d,(56,420),f"{c['side_returns']} / {n} side returns",25,WHITE,True)
    lines(d,(660,224),'Task: grab and lift a bottle, pour, stop near full or when empty, then set down and release.',25,560,WHITE)
    lines(d,(660,360),'Supervised visual perception; retained motion behavioral cloning + IK; programmed grasp, stop and return.',22,560,MUTED)
    lines(d,(56,545),'Original cup and bottle sizes. Material A. Native task, wrist and cropped cup-detail views.',21,1168,MUTED)
    lines(d,(56,610),'Simulation only; no RL or physical validation. Loss: final outside / ever near table at 20 Hz.',19,1168,MUTED)
    return im
def frame(row,record,ds,group,index,t,speed,part,number,total,hold=None):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im);color=GREEN if row['strict_pass'] else RED
    text(d,(24,17),'AIWORKER / ISAAC SIM / '+part.upper(),14,MUTED,True)
    goal='Fill the cup' if row['expected_stop']=='cup_near_full' else 'Empty the bottle'
    text(d,(24,45),f"Attempt {int(row['name'].split('_')[-1])+1:02} / {6 if part=='development' else 30:02}   {goal}",25,WHITE,True,max_width=1040)
    d.rounded_rectangle((1090,22,1256,69),radius=9,fill=color)
    text(d,(1173,46),'PASS' if row['strict_pass'] else 'FAIL',25,BG,True,max_width=150,anchor='mm')
    for camera,x,size,label in (('task',24,512,'TASK CAMERA'),('cup_detail',568,320,'CUP DETAIL / CROPPED'),('wrist',920,320,'WRIST CAMERA')):
        text(d,(x,88),label,14,MUTED,True);image=picture(ds[group+'/'+camera+'_image'],index)
        if camera=='cup_detail':
            u,v=record['cameras'][camera]['rim_center_uv'];box=(round(u-160),round(v-80),round(u+160),round(v+240))
            assert box[0]>=0 and box[1]>=0 and box[2]<=image.width and box[3]<=image.height
            image=image.crop(box)
        im.paste(image.resize((size,size),Image.Resampling.LANCZOS),(x,112))
    title,detail=verdict(row)
    if not row['strict_pass'] and row['zero_spill'] and row['stop_reason']==row['expected_stop']:
        if 'side_return_unconfirmed' in row['failure_reasons']:
            title='Side return failed';detail='The bottle did not meet the continuous sideways uprighting and placement checks.'
        elif 'insufficient_stopping_margin' in row['failure_reasons']:
            title='Stopping margin too small';detail=f"Peak sampled level: {100*row['side_return']['peak_surface_height_fraction']:.1f}% of rim; limit 98%."

    if row['strict_pass'] and row['expected_stop']=='bottle_empty':
        title='Task passed'
        detail=f"Empty threshold met: {row['source_remaining_ml']:.2f} nominal mL left. Correct stop and return; zero measured loss."
    text(d,(568,460),title,24,color,True,max_width=672)
    y=lines(d,(568,500),detail,19,672,MUTED);assert y<594,(row['name'],y)
    text(d,(568,605),phase_label(ds,group,index),20,WHITE,max_width=672)
    text(d,(24,645),f'SIM {t:05.2f} s',18,WHITE,True,max_width=200)
    text(d,(228,645),hold or f'{speed:g}x playback',18,MUTED,max_width=310)
    text(d,(568,645),'Stop near full' if row['expected_stop']=='cup_near_full' else 'Stop when bottle is empty',18,MUTED,max_width=672)
    text(d,(24,684),'Complete native attempt / simulation',16,MUTED,max_width=510)
    text(d,(568,684),'All successes and failures retained',16,MUTED,max_width=672)
    return im
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--preview',action='store_true');p.add_argument('--batch',type=int,default=0);a=p.parse_args()
    stem=f'{a.part}-r{a.round}'
    report=read(E/(stem+'-outcomes.json'));audit=read(E/(stem+'-audit.json'))
    assert report['status']=='complete' and len(report['cases'])==(6 if a.part=='development' else 30) and not audit['started_incomplete']
    selection=read(E/f'candidate-r{a.round}.json');assert selection['checkpoint_sha256']==report['checkpoint_sha256']
    indexed={r['name']:r for r in audit['records']};rows=sorted(report['cases'],key=lambda r:r['name'])
    outstem=stem+(f'-part{a.batch}' if a.part=='test' else '')
    if a.part=='test':
        assert 1<=a.batch<=3;rows=rows[(a.batch-1)*10:a.batch*10]
    folder=E/(outstem+'-video-preview');folder.mkdir(exist_ok=True)
    opening=card(report,a.part);opening.save(folder/'opening.png')
    output=E/(outstem+'.mp4');assert a.preview or not output.exists(),'Preserve existing videos'
    proc=None if a.preview else subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    count=0;clips=[]
    def emit(im):
        nonlocal count
        proc.stdin.write(im.tobytes());count+=1
    try:
        if proc:
            for _ in range(6*FPS):emit(opening)
        for number,row in enumerate(rows,1):
            assert row['strict_pass']==strict_success(row)
            row=dict(row,strict_pass=row['accepted'],failure_reasons=row['acceptance_failures'])
            record=indexed[row['name']];assert record['sha256']==row['source_sha256']
            identity=dict(record,relative_path=str(PurePosixPath(record['path']).relative_to('/workspace/outputs/fountain')))
            speed=2 if row['accepted'] else 1
            with verified_recording(identity) as path,h5py.File(path) as f:
                assert f.attrs['status']=='complete' and json.loads(f.attrs['metrics'])['policy']['checkpoint_sha256']==report['checkpoint_sha256']
                ds=f['data/demo_0'];times=ds['next_time'][:];start=count;schedule=[]
                for label,i in (('initial',0),('middle',len(times)//2),('final',len(times)-1)):
                    frame(row,record,ds,'next_obs',i,float(times[i]),speed,a.part,number,len(rows)).save(folder/(row['name']+'-'+label+'.png'))
                if a.preview:continue
                schedule=[('obs',0)]*FPS+[('next_obs',i) for i in range(0,len(times),speed)]+[('next_obs',len(times)-1)]*(2*FPS)
                for j,(group,i) in enumerate(schedule):
                    hold='Initial layout' if j<FPS else 'Final frame held' if j>=len(schedule)-2*FPS else None
                    emit(frame(row,record,ds,group,i,float(ds['time' if group=='obs' else 'next_time'][i]),speed,a.part,number,len(rows),hold))
                clips.append(dict(name=row['name'],strict_pass=row['strict_pass'],failure_reasons=row['failure_reasons'],source_sha256=record['sha256'],start_frame=start,end_frame_exclusive=count,playback_speed=speed,native_frames=len(times),mapping=schedule))
            print(row['name'],row['strict_pass'],flush=True)
        if a.preview:return
        for _ in range(5*FPS):emit(opening)
        proc.stdin.close();assert proc.wait()==0
    except BaseException:
        if proc:proc.kill();proc.wait()
        raise
    result=dict(part=a.part,batch=a.batch,outcomes_stem=stem,video=str(output),sha256=sha(output),bytes=output.stat().st_size,checkpoint_sha256=report['checkpoint_sha256'],report_sha256=sha(E/(stem+'-outcomes.json')),audit_sha256=sha(E/(stem+'-audit.json')),helper_sha256=sha(Path(__file__)),frames=count,fps=FPS,duration_s=count/FPS,resolution=list(SIZE),clips=clips,visual_review='pending',scope='All declared native attempts in this part at labeled 1x/2x speed, with initial/final holds and no internal cuts or generated camera frames. Cup-detail display window uses known fixture center only for human review; the controller uses image-localized position.')
    (E/(outstem+'-video.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='clips'}))
if __name__=='__main__':main()
