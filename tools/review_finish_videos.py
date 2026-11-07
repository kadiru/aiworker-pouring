"""Decode encoded test reels and compare displayed views with native frames."""
import argparse,io,json,subprocess
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw
from render_labeled_benchmark import font,sha

PROJECT=Path(__file__).resolve().parents[1];E=PROJECT/'evidence/finish-A';DATA=None
from finish_media_source import verified_recording


def main(part):
    manifest=json.loads((E/f'video-part-{part}-manifest.json').read_text());video=Path(manifest['video'])
    assert sha(video)==manifest['sha256'];dest=E/f'video-review-{part}';dest.mkdir(exist_ok=False)
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    stream=probe['streams'][0]
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and stream['width']==1280 and stream['height']==720
    assert int(stream['nb_frames'])==manifest['frames'] and stream['avg_frame_rate']=='20/1'
    requests=[dict(name='opening',stage='card',frame=30)]+manifest['review_frames']+[dict(name='closing',stage='card',frame=manifest['frames']-30)]
    assert len(requests)==32 and len(manifest['clips'])==10
    expression="select='"+'+'.join(f"eq(n,{r['frame']})" for r in requests)+"'"
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-vf',expression,'-fps_mode','vfr',str(dest/'encoded-%03d.png')],check=True)
    comparisons=[];clips={r['name']:r for r in manifest['clips']}
    integrity=json.loads((E/'recording-integrity.json').read_text());indexed={r['name']:r for r in integrity['files']}
    for i,request in enumerate(requests,1):
        path=dest/f'encoded-{i:03d}.png';assert path.exists();request['path']=str(path);request['sha256']=sha(path)
    for name,clip in clips.items():
        identity=indexed[name];assert clip['source_sha256']==identity['sha256']
        with verified_recording(identity) as source,h5py.File(source) as f:
            ds=f['data/demo_0']
            for request in (r for r in requests if r['name']==name):
                group,index=clip['output_frame_to_native_observation'][request['frame']-clip['start_frame']]
                encoded=Image.open(request['path']).convert('RGB')
                for cam,x,y,size in [('overview',24,112,512),('task',568,112,320),('wrist',920,112,320)]:
                    native=Image.open(io.BytesIO(ds[group+'/'+cam+'_image'][index].tobytes())).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
                    a=np.asarray(native,dtype=float);b=np.asarray(encoded.crop((x,y,x+size,y+size)),dtype=float)
                    error=float(abs(a-b).mean());assert error<2.5,(request,cam,error)
                    comparisons.append(dict(name=request['name'],stage=request['stage'],camera=cam,mean_absolute_pixel_error=error))
    # Two complete initial/middle/final triptychs per page. Native frame views
    # remain in their encoded aspect ratio; no claim of manual review yet.
    footage=requests[1:-1]
    for page in range(5):
        sheet=Image.new('RGB',(1280,544),'#101820');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(footage[page*6:(page+1)*6]):
            x=(j%3)*426;y=(j//3)*272
            sheet.paste(Image.open(r['path']).resize((426,240),Image.Resampling.LANCZOS),(x,y))
            draw.text((x+6,y+244),r['name']+' / '+r['stage'],font=font(13),fill='white')
        sheet.save(dest/f'contact-{page+1}.jpg',quality=95)
    report=dict(part=part,video_sha256=manifest['sha256'],full_decode_passed=True,frames=manifest['frames'],
        duration_s=float(probe['format']['duration']),encoded_review_frames=requests,
        camera_comparisons=comparisons,maximum_mean_absolute_pixel_error=max(r['mean_absolute_pixel_error'] for r in comparisons),
        all_ten_have_initial_middle_final=True,visual_review='pending')
    (E/f'video-part-{part}-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(part=part,full_decode=True,review_frames=32,comparisons=len(comparisons),visual_review='pending')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=(1,2,3),required=True);main(p.parse_args().part)
