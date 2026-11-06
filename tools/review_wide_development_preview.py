"""Decode a development preview and compare selected views to native images."""
import argparse
import io
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image
from render_labeled_benchmark import sha

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'evidence/wide-A'
DATA = ROOT/'data/material-A-wide'


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--index',type=Path,required=True);a=p.parse_args()
    assert a.name.startswith('XW30_A_dev_')
    m=json.loads((E/(a.name+'-preview.json')).read_text());video=Path(m['video'])
    assert sha(video)==m['sha256']
    identity=next(r for r in json.loads(a.index.read_text())['files'] if r['name']==a.name)
    source=DATA/identity['relative_path'];assert sha(source)==m['source_sha256']==identity['sha256']
    decisions=json.loads((source.parent/'policy-decisions.json').read_text())
    stop=next(i for i,r in enumerate(decisions) if r['stop_latched'])
    retreat=next(i for i,r in enumerate(decisions) if r['mode']=='Retreating')
    indices=[10,20+stop//m['playback_speed'],20+retreat//m['playback_speed'],m['frames']-20]
    dest=E/(a.name+'-preview-review');dest.mkdir(exist_ok=False)
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    stream=probe['streams'][0]
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and stream['avg_frame_rate']=='20/1'
    assert (stream['width'],stream['height'],int(stream['nb_frames']))==(1280,720,m['frames'])
    expression="select='"+'+'.join(f'eq(n,{i})' for i in indices)+"'"
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-vf',expression,'-fps_mode','vfr',str(dest/'encoded-%02d.png')],check=True)
    comparisons=[];sheet=Image.new('RGB',(1440,810),'#101820')
    with h5py.File(source) as f:
        ds=f['data/demo_0']
        for j,frame in enumerate(indices,1):
            im=Image.open(dest/f'encoded-{j:02d}.png').convert('RGB');group,index=m['mapping'][frame]
            for cam,x,y,size in [('overview',24,112,512),('task',568,112,320),('wrist',920,112,320)]:
                native=Image.open(io.BytesIO(ds[group+'/'+cam+'_image'][index].tobytes())).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
                error=float(abs(np.asarray(native,dtype=float)-np.asarray(im.crop((x,y,x+size,y+size)),dtype=float)).mean())
                assert error<2.5
                comparisons.append(dict(frame=frame,camera=cam,mean_absolute_pixel_error=error))
            sheet.paste(im.resize((720,405),Image.Resampling.LANCZOS),(((j-1)%2)*720,((j-1)//2)*405))
    sheet.save(dest/'review.jpg',quality=95)
    report=dict(video_sha256=m['sha256'],full_decode_passed=True,frames=m['frames'],duration_s=float(probe['format']['duration']),
        review_frame_numbers=indices,camera_comparisons=comparisons,maximum_mean_absolute_pixel_error=max(r['mean_absolute_pixel_error'] for r in comparisons),
        visual_review='pending',scope='Four selected encoded frames: initial, stop, retreat start and final. Full decoding is automatic; manual selected-frame review remains required.')
    (E/(a.name+'-preview-review.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='camera_comparisons'}))


if __name__=='__main__':main()
