"""Three clean reels covering all thirty fresh native autonomous tests.

Local CPU composition. Complete attempts at labeled 1x/2x, no synthesized
camera frames or success-only selection. Requires verified local recordings.
"""
import argparse,io,json,subprocess,sys
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw
from render_labeled_benchmark import font,text,sha
from render_layout_variety import lines,layout_map,verdict as prior_verdict

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/wide-A';DATA=ROOT/'data/material-A-wide'
FPS=20;SIZE=(1280,720);BG='#101820';WHITE='#F3F6F7';MUTED='#B3C0CC';GREEN='#67D6B1';RED='#FFAAA3'
sys.path.insert(0,str(ROOT/'runtime/src'))
from pour_wide_report import strict_success


def read(p):return json.loads(p.read_text())
def picture(ds,i):return Image.open(io.BytesIO(ds[i].tobytes())).convert('RGB')


def verdict(row):
    if row['strict_pass']:return prior_verdict(row)
    if (row['stop_reason'] in ('localization_unconfirmed','approach_unconfirmed') or not row['grasp_verified']
        or not row['visual_grasp_confirmed'] or not row['zero_spill'] or row['sustained_grasp_fraction']<.8
        or not row['completed_return_release_sequence']):return prior_verdict(row)
    if row['stop_reason'] in ('trajectory_unreachable','return_planning_failed'):
        return 'Motion planning failed','A required joint path could not be confirmed. The controller aborted the task.'
    if row['stop_reason']!=row['expected_stop']:
        return 'Wrong stop condition','The controller did not confirm the required full-cup or empty-bottle condition.'
    if row['expected_stop']=='bottle_empty' and row['source_remaining_ml']>max(.25,.01*row['source_initial_ml']):
        return 'Bottle not empty',f"{row['source_remaining_ml']:.2f} nominal mL remained above the allowed empty threshold."
    if row['expected_stop']=='cup_near_full' and row['cup_surface_height_fraction']<.9:return prior_verdict(row)
    return 'Task checks failed','The attempt did not meet every required grasp, transfer, stop and return check.'


def frame(row,ds,group,index,t,speed,hold=None):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im);color=GREEN if row['strict_pass'] else RED
    text(d,(24,17),'AIWORKER  /  ISAAC SIM  /  FRESH AUTONOMOUS TESTS',14,MUTED,True)
    text(d,(24,45),f"{row['test_index']+1:02} / 30   {row['layout_label']}",26,WHITE,True,max_width=1010)
    d.rounded_rectangle((1090,22,1256,69),radius=9,fill=color)
    text(d,(1173,46),'PASS' if row['strict_pass'] else 'FAIL',25,BG,True,max_width=150,anchor='mm')
    for name,x,y,size,label in [('overview',24,112,512,'TABLE OVERVIEW'),('task',568,112,320,'TASK CAMERA'),('wrist',920,112,320,'WRIST CAMERA')]:
        text(d,(x,88),label,14,MUTED,True)
        im.paste(picture(ds[group+'/'+name+'_image'],index).resize((size,size),Image.Resampling.LANCZOS),(x,y))
    layout_map(d,row)
    title,detail=verdict(row)
    text(d,(920,460),'FINAL OUTCOME',14,MUTED,True)
    y=lines(d,(920,490),title,24,320,color,True);y=lines(d,(920,y+12),detail,17,320,MUTED)
    assert y<651,(row['name'],y)
    text(d,(24,645),f'SIM {t:05.2f} s',18,WHITE,True,max_width=200)
    text(d,(228,645),hold or f'{speed:g}x playback',18,MUTED,max_width=305)
    text(d,(24,681),'Stop near full' if row['expected_stop']=='cup_near_full' else 'Stop when empty',16,MUTED,max_width=500)
    text(d,(568,674),'One frozen controller  ·  Each test runs once',16,MUTED,max_width=670)
    return im


def card(report,model,part,closing=False):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im);c=report['counts']
    text(d,(70,55),'AIWORKER  /  ISAAC SIM',18,MUTED,True)
    text(d,(70,107),f'Wider placements. Exact swaps.  Part {part}/3',36,WHITE,True)
    d.line((70,174,1210,174),fill='#33414F',width=2)
    text(d,(70,211),f"{c['strict_passes']} / 30",78,WHITE,True)
    text(d,(70,308),'complete tasks passed',24,MUTED)
    text(d,(70,365),f"{c['zero_measured_loss']} / 30",48,GREEN,True)
    text(d,(70,428),'attempts with zero measured loss',21,MUTED)
    if not closing:
        lines(d,(610,209),'Grasp → lift → pour → stop → return',29,570,WHITE,True)
        lines(d,(610,278),'Stop at ≥90% rim height or when the bottle is empty. The goal includes zero measured spill.',23,570,MUTED)
        lines(d,(610,393),'All 30 fresh attempts appear across three reels. Paired layouts exchange bottle and cup positions.',22,570,WHITE)
    else:
        lines(d,(610,210),'Supervised learning on the RTX 4070',28,570,WHITE,True)
        training=model['training'];source=training['empty_readout_fit']
        lines(d,(610,279),f"RGB localization · {training['localization_template_episodes']} training / {training['localization_validation_episodes']} validation references",22,570,MUTED)
        lines(d,(610,361),f"Motion behavioral cloning + IK constraints. Settled-height ridge regression; {source['learned_parameters']:,}-parameter wrist CNN for empty detection.",22,570,WHITE)
        lines(d,(610,470),'Programmed grasp, stop and return logic. No reinforcement learning; no physical robot test.',21,570,MUTED)
    lines(d,(70,585),'Native task, wrist and overview cameras. Complete runs at labeled playback speed; no generated camera footage.',19,1140,MUTED)
    lines(d,(70,643),'Spill is measured at 20 Hz in simulation. Transient returning droplets are allowed by this metric.',18,1140,MUTED)
    return im


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',type=int,choices=(1,2,3),required=True);parser.add_argument('--preview',action='store_true')
    args=parser.parse_args();report=read(E/'final-benchmark.json');model=read(E/'model-summary.json')
    assert report['status']=='complete' and len(report['cases'])==30
    integrity=read(E/'recording-integrity.json');indexed={r['name']:r for r in integrity['files']}
    all_rows=sorted(report['cases'],key=lambda r:r['name'])
    for i,r in enumerate(all_rows):
        assert r['strict_pass']==strict_success(r)
        r['test_index']=i;x,y,z=r['cup_world'];a,b=r['source_xy'];r['bottle_world_xy']=[x-b,y+a]
    rows=all_rows[(args.part-1)*10:args.part*10];paths=[]
    for r in rows:
        identity=indexed[r['name']];path=DATA/identity['relative_path']
        assert path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256'] and identity['valid']
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete' and f.attrs['evidence_kind']=='learned_visual_policy'
            spec=json.loads(f.attrs['spec']);assert spec['name']==r['name'] and spec['protocol']==17
            assert json.loads(f.attrs['metrics'])['policy']['checkpoint_sha256']==report['checkpoint_sha256']
        paths.append(path)
    out=E/f'thirty-wide-tests-part-{args.part}.mp4';assert not out.exists(),'Preserve completed exports'
    if args.preview:
        dest=E/f'video-preview-{args.part}';dest.mkdir(exist_ok=True)
        card(report,model,args.part).save(dest/'opening.png');card(report,model,args.part,True).save(dest/'closing.png')
        for r,path in zip(rows,paths):
            with h5py.File(path) as f:
                ds=f['data/demo_0'];i=len(ds['actions'])-1
                frame(r,ds,'next_obs',i,float(ds['next_time'][i]),2,'Final frame held').save(dest/(r['name']+'.png'))
        print(dest);return
    proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
        '-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    count=0;clips=[];review=[]
    def emit(im):
        nonlocal count
        proc.stdin.write(im.tobytes());count+=1
    try:
        opening=card(report,model,args.part)
        for _ in range(6*FPS):emit(opening)
        for r,path in zip(rows,paths):
            with h5py.File(path) as f:
                ds=f['data/demo_0'];times=ds['next_time'][:];speed=1 if not r['zero_spill'] else 2
                start=count;schedule=[]
                first=frame(r,ds,'obs',0,float(ds['time'][0]),speed,'Initial layout')
                for _ in range(FPS):emit(first);schedule.append(['obs',0])
                for i in range(0,len(times),speed):
                    emit(frame(r,ds,'next_obs',i,float(times[i]),speed));schedule.append(['next_obs',i])
                final=frame(r,ds,'next_obs',len(times)-1,float(times[-1]),speed,'Final frame held')
                for _ in range(2*FPS):emit(final);schedule.append(['next_obs',len(times)-1])
                review.extend(dict(name=r['name'],stage=stage,frame=at) for stage,at in
                    [('initial',start+FPS//2),('middle',start+FPS+len(times)//speed//2),('final',count-FPS)])
                clips.append(dict(name=r['name'],strict_pass=r['strict_pass'],failure_reasons=r['failure_reasons'],
                    source=str(path.relative_to(DATA)),source_sha256=sha(path),start_frame=start,end_frame_exclusive=count,
                    playback_speed=speed,native_sim_seconds=float(times[-1]-ds['time'][0]),native_frames=len(times),
                    output_frame_to_native_observation=schedule))
            print(r['name'],r['strict_pass'],flush=True)
        ending=card(report,model,args.part,True)
        for _ in range(8*FPS):emit(ending)
        proc.stdin.close();assert proc.wait()==0
    except BaseException:
        proc.kill();proc.wait();raise
    meta=dict(video=str(out),sha256=sha(out),bytes=out.stat().st_size,part=args.part,frames=count,fps=FPS,resolution=list(SIZE),
        duration_s=count/FPS,report_sha256=sha(E/'final-benchmark.json'),clips=clips,review_frames=review,
        edits='Complete attempts at labeled 1x/2x with initial/final holds; no internal cuts, interpolation or generated camera frames.')
    (E/f'video-part-{args.part}-manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k not in ('clips','review_frames')},indent=2))

if __name__=='__main__':main()
