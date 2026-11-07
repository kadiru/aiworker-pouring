"""One complete, explicitly labeled development recording; not a test score."""
import argparse,json,subprocess,sys
from pathlib import Path
import h5py
from PIL import ImageDraw
from render_finish_videos import frame,BG,WHITE,MUTED,FPS,SIZE,DATA,E
from render_labeled_benchmark import text,sha

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons
from validate_pour_recording import validate
from finish_media_source import verified_recording
DATA=Path(__file__).resolve().parents[1]/"data/material-A-finish"


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--index',type=Path,required=True);a=p.parse_args()
    assert a.name.startswith('FN30_A_dev_')
    index=json.loads(a.index.read_text());item=next(r for r in index['files'] if r['name']==a.name)
    folder=(DATA/item['relative_path']).parent
    with verified_recording(item) as path:
        for r in index['sidecar_files']:
            if Path(r['relative_path']).parent==Path(item['relative_path']).parent:assert sha(DATA/r['relative_path'])==r['sha256']
        check=validate(path);assert check['valid']
        spec=json.loads((folder/'parameters.json').read_text());checkpoint=check['metrics']['policy']['checkpoint_sha256']
        row=completed_row(folder,checkpoint);row.update(strict_pass=strict_success(row),failure_reasons=reasons(row),test_index=0,layout_label=spec['layout_label'])
        speed=1 if not row['zero_spill'] else 2
        x,y,_=row['cup_world'];dx,dy=row['source_xy'];row['bottle_world_xy']=[x-dy,y+dx]
        out=E/(a.name+'-preview.mp4');assert not out.exists()
        def draw(ds,group,i,t,hold=None):
            im=frame(row,ds,group,i,t,speed,hold);d=ImageDraw.Draw(im)
            d.rectangle((0,0,1080,81),fill=BG)
            text(d,(24,17),'AIWORKER  /  ISAAC SIM  /  DEVELOPMENT',14,MUTED,True)
            text(d,(24,45),row['layout_label'],26,WHITE,True,max_width=1040)
            d.rectangle((560,665,1279,719),fill=BG)
            text(d,(568,674),'Development only · excluded from the 30 fresh tests',16,MUTED,max_width=680)
            return im
        proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
        mapping=[]
        try:
            with h5py.File(path) as f:
                d=f['data/demo_0'];times=d['next_time'][:]
                schedule=[('obs',0)]*FPS+[('next_obs',i) for i in range(0,len(times),speed)]+[('next_obs',len(times)-1)]*(2*FPS)
                for j,(group,i) in enumerate(schedule):
                    im=draw(d,group,i,float(d['time' if group=='obs' else 'next_time'][i]),'Initial layout' if j<FPS else 'Final frame held' if j>=len(schedule)-2*FPS else None)
                    proc.stdin.write(im.tobytes());mapping.append([group,i])
            proc.stdin.close();assert proc.wait()==0
        except BaseException:
            proc.kill();proc.wait();raise
        report=dict(name=a.name,video=str(out),sha256=sha(out),source_sha256=sha(path),checkpoint_sha256=checkpoint,strict_pass=row['strict_pass'],failure_reasons=row['failure_reasons'],cup_surface_height_fraction=row['cup_surface_height_fraction'],source_remaining_ml=row['source_remaining_ml'],final_outside_ml=row['final_outside_ml'],ever_below_worktop_ml=row['ever_below_worktop_ml'],fps=FPS,resolution=list(SIZE),frames=len(mapping),duration_s=len(mapping)/FPS,mapping=mapping,playback_speed=speed,scope=f'Complete native development attempt at labeled {speed}x; not part of reserved-test results. No generated camera footage.',visual_review='pending')
        (E/(a.name+'-preview.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='mapping'}))



if __name__=='__main__':main()
