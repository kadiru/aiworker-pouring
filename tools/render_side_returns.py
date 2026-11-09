"""Normal-speed return segments from the same audited autonomous attempts."""
import argparse,json,subprocess
from pathlib import Path,PurePosixPath
import h5py
from PIL import Image,ImageDraw
from finish_media_source import verified_recording
from render_side_tasks import frame,read,sha,text,lines,E,FPS,SIZE,BG,WHITE,MUTED

def card(report):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im)
    text(d,(56,48),'AIWORKER / ISAAC SIM / RETURN SEGMENTS',18,MUTED,True)
    text(d,(56,118),'Rotate upright while moving aside.',38,WHITE,True)
    lines(d,(56,214),'Normal simulation speed: 1x. Each segment begins at the autonomous stop and continues through set-down, release and hand withdrawal.',26,1120,WHITE)
    lines(d,(56,356),'Robot-right is preferred; robot-left is used when the planned right-side placement is unavailable.',25,1120,MUTED)
    lines(d,(56,468),'Native footage from the same scored attempts. PASS / FAIL labels describe the entire task, including pouring and placement.',23,1120,MUTED)
    c=report['counts'];text(d,(56,620),f"{c['accepted']} / {c['completed']} complete tasks accepted. All return segments retained.",22,WHITE,True)
    return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);a=p.parse_args()
    base=f'development-r{a.round}';stem=base+'-returns';report=read(E/(base+'-outcomes.json'));audit=read(E/(base+'-audit.json'))
    assert report['status']=='complete' and len(report['cases'])==6 and not audit['started_incomplete']
    assert report['checkpoint_sha256']==read(E/f'candidate-r{a.round}.json')['checkpoint_sha256']
    indexed={r['name']:r for r in audit['records']};output=E/(stem+'.mp4');assert not output.exists()
    proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    count=0;clips=[]
    def emit(im):
        nonlocal count
        proc.stdin.write(im.tobytes());count+=1
    try:
        for _ in range(5*FPS):emit(card(report))
        for number,original in enumerate(sorted(report['cases'],key=lambda r:r['name']),1):
            row=dict(original,strict_pass=original['accepted'],failure_reasons=original['acceptance_failures'])
            r=indexed[row['name']];assert r['sha256']==row['source_sha256']
            identity=dict(r,relative_path=str(PurePosixPath(r['path']).relative_to('/workspace/outputs/fountain')))
            with verified_recording(identity) as path,h5py.File(path) as f:
                assert f.attrs['status']=='complete'
                assert json.loads(f.attrs['metrics'])['policy']['checkpoint_sha256']==report['checkpoint_sha256']
                ds=f['data/demo_0'];times=ds['next_time'][:]
                begin=original['withdrawal']['stage_starts_s']['Returning upright'];assert begin is not None
                start_index=next(i for i,t in enumerate(times) if t>=begin-1e-5)
                mapping=[('next_obs',i) for i in range(start_index,len(times))]+[('next_obs',len(times)-1)]*FPS
                start=count
                for j,(group,i) in enumerate(mapping):
                    im=frame(row,r,ds,group,i,float(times[i]),1,'development',number,6,'Final frame held' if j>=len(mapping)-FPS else None)
                    d=ImageDraw.Draw(im);d.rectangle((0,0,1080,37),fill=BG)
                    text(d,(24,17),'AIWORKER / RETURN SEGMENT / ROBOT '+row['side_return']['side'].upper(),14,MUTED,True)
                    d.rectangle((0,674,550,720),fill=BG);text(d,(24,684),'Native return segment / normal simulation speed',16,MUTED,max_width=510)
                    emit(im)
                clips.append(dict(name=row['name'],strict_pass=original['strict_pass'],accepted=original['accepted'],acceptance_failures=original['acceptance_failures'],source_sha256=r['sha256'],start_frame=start,end_frame_exclusive=count,playback_speed=1,native_frames=len(times),mapping=mapping,segment_start_sim_seconds=float(times[start_index])))
        for _ in range(4*FPS):emit(card(report))
        proc.stdin.close();assert proc.wait()==0
    except BaseException:
        proc.kill();proc.wait();raise
    result=dict(part='development',outcomes_stem=base,video=str(output),sha256=sha(output),bytes=output.stat().st_size,checkpoint_sha256=report['checkpoint_sha256'],report_sha256=sha(E/(base+'-outcomes.json')),audit_sha256=sha(E/(base+'-audit.json')),helper_sha256=sha(Path(__file__)),frames=count,fps=FPS,duration_s=count/FPS,resolution=list(SIZE),clips=clips,visual_review='pending',scope='All six native return segments from autonomous stop through final holding pose, at 1x with no internal cuts. These excerpts omit grasp/pour and supplement the full-attempt reel. Labels use complete-task acceptance, including failures.')
    (E/(stem+'-video.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='clips'}))
if __name__=='__main__':main()
