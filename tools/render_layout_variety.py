"""Show all twelve native autonomous runs, outcome labels and initial XY maps.

Mac CPU composition only. No interpolation or synthesized simulation frames.
Every trial is retained from start to finish, at explicitly labeled 1x/2x speed.
"""
import io
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw
from render_labeled_benchmark import font, text, sha

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/layout-variety-A';DATA=ROOT/'data/material-A-layout-variety'
OUT=E/'fourteen-autonomous-layouts.mp4'
FPS=20;SIZE=(1280,720)
BG='#101820';WHITE='#F3F6F7';MUTED='#B3C0CC';GREEN='#67D6B1';RED='#FFAAA3'
BLUE='#78B7F3';PINK='#ED9CBD'

def read(p):return json.loads(p.read_text())
def image(ds,i):return Image.open(io.BytesIO(ds[i].tobytes())).convert('RGB')

def lines(draw,xy,value,size,width,color=WHITE,bold=False,leading=1.3):
    words=value.split();line='';y=xy[1]
    for word in words:
        trial=(line+' '+word).strip()
        if draw.textlength(trial,font=font(size,bold))>width and line:
            text(draw,(xy[0],y),line,size,color,bold,max_width=width);y+=size*leading;line=word
        else:line=trial
    if line:text(draw,(xy[0],y),line,size,color,bold,max_width=width);y+=size*leading
    return y

def verdict(row):
    if row.get('setup_flag'):
        if row['initial_bottle_hand_contact']:
            return 'Initial hand contact',('Bottle/finger contact at the start; liquid escaped. Localization also rejected.' if not row['zero_spill'] else 'Bottle/finger contact at the start. Localization also rejected; no confirmed grasp.')
        return 'Initial overlap flagged','Cup/finger overlap in geometry audit. Localization also rejected; no pour.'
    if row['strict_pass']:
        title='Full task passed'
        detail=('Liquid height: '+f"{100*row['cup_surface_height_fraction']:.1f}%"+' of rim. Correct stop and return.' if row['expected_stop']=='cup_near_full'
                else 'Bottle emptied; correct stop and return.')
        return title,detail+' Zero measured loss.'
    reason=row['stop_reason']
    if reason=='localization_unconfirmed':
        if not row['zero_spill']:
            return 'Localization + spill','No confirmed grasp. Liquid also escaped during the attempt.'
        return 'Localization rejected','No confirmed grasp or pour. The controller did not accept the object positions.'
    if reason=='approach_unconfirmed':return 'Grasp pose not reached','The arm could not complete its approach. The controller aborted.'
    if not row['visual_grasp_confirmed'] or not row['grasp_verified']:return 'Grasp not confirmed','The controller failed to establish a verified hold on the bottle.'
    if not row['zero_spill']:return 'Spill detected','Liquid escaped the vessels. The zero-spill goal failed.'
    if row['sustained_grasp_fraction']<.8:return 'Grip not sustained',f"Two-finger contact on {100*row['sustained_grasp_fraction']:.1f}% of tipping samples; 80% required."
    if not row['completed_return_release_sequence']:return 'Stop / return failed','The controller did not complete the stop, upright return and release.'
    return 'Cup underfilled',f"Final liquid height: {100*row['cup_surface_height_fraction']:.1f}% of rim. Required: at least 90%."

def layout_map(d,row):
    text(d,(568,460),'PLANNED START',14,MUTED,True)
    # True XY aspect ratio, robot on the lower-X (left) side.
    x0,y0=598,492;scale=310
    w,h=.38*scale,.46*scale
    d.rounded_rectangle((x0,y0,x0+w,y0+h),radius=5,fill='#273541',outline='#687887',width=1)
    def xy(x,y):return x0+(x-.305)*scale,y0+(.03-y)*scale
    for coords,color,letter in [(row['bottle_world_xy'],BLUE,'B'),(row['cup_world'][:2],PINK,'C')]:
        x,y=xy(*coords);d.ellipse((x-8,y-8,x+8,y+8),fill=color)
        text(d,(x,y),letter,10,BG,True,max_width=12,anchor='mm')
    d.line((580,y0+34,580,y0+h-34),fill='#7B8B99',width=3)
    text(d,(744,501),'B  Bottle',18,BLUE,True,max_width=145)
    text(d,(744,533),'C  Cup',18,PINK,True,max_width=145)
    text(d,(744,580),'Robot at left',15,MUTED,max_width=145)
    text(d,(744,606),'Map to scale',15,MUTED,max_width=145)

def frame(row,ds,group,index,t,speed,hold=None):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im)
    number=row['layout_index']+1
    text(d,(24,17),'AIWORKER  /  ISAAC SIM  /  AUTONOMOUS LAYOUT TESTS',14,MUTED,True)
    text(d,(24,45),f"{number:02} / 14   {row['layout_label']}",26,WHITE,True,max_width=1010)
    color=GREEN if row['strict_pass'] else RED
    d.rounded_rectangle((1090,22,1256,69),radius=9,fill=color)
    badge='SETUP FAIL' if row.get('setup_flag') else 'PASS' if row['strict_pass'] else 'FAIL'
    text(d,(1173,46),badge,21 if row.get('setup_flag') else 25,BG,True,max_width=150,anchor='mm')
    text(d,(24,88),'TABLE OVERVIEW',14,MUTED,True)
    text(d,(568,88),'TASK CAMERA',14,MUTED,True)
    text(d,(920,88),'WRIST CAMERA',14,MUTED,True)
    for name,x,y,size in [('overview',24,112,512),('task',568,112,320),('wrist',920,112,320)]:
        native=image(ds[group+'/'+name+'_image'],index)
        im.paste(native.resize((size,size),Image.Resampling.LANCZOS),(x,y))
    layout_map(d,row)
    title,detail=verdict(row)
    text(d,(920,460),'FINAL OUTCOME',14,MUTED,True)
    y=lines(d,(920,490),title,24,320,color,True)
    y=lines(d,(920,y+12),detail,17,320,MUTED)
    assert y<651,(row['name'],y)
    text(d,(24,645),f'SIM  {t:05.2f} s',18,WHITE,True,max_width=200)
    playback=hold if hold else f'{speed:g}x playback'
    text(d,(228,645),playback,18,MUTED,max_width=305)
    target='Stop near full' if row['expected_stop']=='cup_near_full' else 'Stop when empty'
    text(d,(24,681),target,16,MUTED,max_width=500)
    text(d,(568,674),'Same controller throughout  ·  No retries or teacher fallback',16,MUTED,max_width=670)
    return im

def card(report,closing=False):
    im=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(im);c=report['counts']
    text(d,(70,60),'AIWORKER  /  ISAAC SIM',18,MUTED,True)
    text(d,(70,120),'14 layouts. One unchanged controller.',42,WHITE,True)
    d.line((70,190,1210,190),fill='#33414F',width=2)
    text(d,(70,235),f"{c['strict_passes']} / 14",86,WHITE,True)
    text(d,(70,342),'complete tasks passed',26,MUTED)
    if closing:
        lines(d,(610,237),f"{c['setup_flags']} initial-layout issues. {c['unflagged_starting_layouts']} starts without those flags.",30,560,WHITE,True)
        other=[r for r in report['cases'] if not r['strict_pass'] and not r.get('setup_flag')]
        detail='Other failed layouts: localization rejected. No-pour attempts are not passes.' if all(r['stop_reason']=='localization_unconfirmed' for r in other) else 'Each failure is labeled with its cause. No-pour attempts are not passes.'
        lines(d,(610,355),detail,24,560,MUTED)
        text(d,(70,490),'All fourteen native attempts are included.',28,WHITE,True)
        lines(d,(70,548),'Larger placements expose the current controller’s limits. No retraining was performed for this video.',24,1110,MUTED)
    else:
        lines(d,(610,237),'Five exact bottle/cup swaps, side-by-side placements, diagonals and table extremes.',28,560,WHITE,True)
        lines(d,(610,390),f"Every attempt shown, including {c['setup_flags']} flagged setups. Speed and outcome labeled.",24,560,MUTED)
        lines(d,(70,515),'PASS requires grasp, pour, correct full/empty stop, upright return and zero measured loss.',25,1120,WHITE)
    text(d,(70,658),'Material A  ·  Frozen v18 controller  ·  Deliberate simulation stress test',17,MUTED)
    return im

def main():
    report=read(E/'final-benchmark.json');integrity=read(E/'recording-integrity.json')
    backup=read(E/'backup-verification.json');assert backup['all_native_hashes_and_sizes_match']
    cases=sorted(report['cases'],key=lambda r:r['layout_index']);assert len(cases)==14 and {r['layout_index'] for r in cases}==set(range(14))
    hashes={f['relative_path']:f for f in integrity['files']}
    frame_count=0;manifest=[];review=[];E.joinpath('video-review').mkdir(exist_ok=True)
    command=['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    def emit(im):
        nonlocal frame_count
        proc.stdin.write(im.tobytes());frame_count+=1
    try:
        for _ in range(3*FPS):emit(card(report))
        for row in cases:
            rel=Path(row['path']).relative_to('/workspace/outputs/fountain')/'episode.hdf5';path=DATA/rel
            identity=hashes[str(rel)];assert path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256']
            start=frame_count;schedule=[]
            with h5py.File(path,'r') as f:
                assert f.attrs['status']=='complete';ds=f['data/demo_0'];times=ds['next_time'][:]
                duration=float(times[-1]-ds['time'][0]);speed=1 if duration<=6 else 2
                first=frame(row,ds,'obs',0,float(ds['time'][0]),speed,'Initial layout')
                for _ in range(FPS):emit(first);schedule.append(['obs',0])
                for i in range(0,len(times),speed):
                    emit(frame(row,ds,'next_obs',i,float(times[i]),speed));schedule.append(['next_obs',i])
                final=frame(row,ds,'next_obs',len(times)-1,float(times[-1]),speed,'Final frame held')
                for _ in range(2*FPS):emit(final);schedule.append(['next_obs',len(times)-1])
                review.extend([dict(name=row['name'],stage='initial',frame=start+FPS//2),
                    dict(name=row['name'],stage='middle',frame=start+FPS+int((len(times)-1)/speed)//2),
                    dict(name=row['name'],stage='final',frame=frame_count-FPS)])
                manifest.append(dict(name=row['name'],layout=row['layout_label'],strict_pass=row['strict_pass'],failure_reasons=row['failure_reasons'],
                    source=str(rel),source_sha256=identity['sha256'],start_frame=start,end_frame_exclusive=frame_count,
                    playback_speed=speed,native_sim_seconds=duration,native_frames=len(times),output_frame_to_native_observation=schedule))
            print(json.dumps(dict(name=row['name'],frames=frame_count-start,strict_pass=row['strict_pass'])),flush=True)
        for _ in range(4*FPS):emit(card(report,True))
        proc.stdin.close();assert proc.wait()==0
    except BaseException:
        proc.kill();proc.wait();raise
    meta=dict(video=str(OUT),sha256=sha(OUT),bytes=OUT.stat().st_size,resolution=list(SIZE),fps=FPS,
        frames=frame_count,duration_s=frame_count/FPS,report_sha256=sha(E/'final-benchmark.json'),
        all_fourteen_included=True,temporal_edits='Complete runs at labeled1x/2x, plus1s initial and2s final holds. No internal cuts or interpolation.',
        image_edits='Resize original overview/task/wrist frames only. Add outcome text and a schematic initial XY map from the frozen fixture plan.',
        cases=manifest,review_frame_requests=review)
    (E/'video-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='cases' and k!='review_frame_requests'}),flush=True)

if __name__=='__main__':main()
