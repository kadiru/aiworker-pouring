"""Full reel decoding and native-frame comparison; manual review stays explicit."""
import argparse,json,subprocess
from pathlib import Path,PurePosixPath
import h5py,numpy as np
from PIL import Image,ImageDraw
from render_labeled_benchmark import font,sha
from render_finish_videos import picture
from finish_media_source import verified_recording
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/side-return-A'
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--batch',type=int,default=0);p.add_argument('--returns-only',action='store_true');a=p.parse_args()
    assert not a.returns_only or (a.part=='development' and a.batch==0)
    base=f'{a.part}-r{a.round}';stem=base+(f'-part{a.batch}' if a.part=='test' else '')+('-returns' if a.returns_only else '');manifest=read(E/(stem+'-video.json'));video=Path(manifest['video'])
    assert sha(video)==manifest['sha256'];dest=E/(stem+'-encoded-review');dest.mkdir(exist_ok=False)
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    stream=probe['streams'][0]
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and [stream['width'],stream['height']]==[1280,720]
    assert int(stream['nb_frames'])==manifest['frames'] and stream['avg_frame_rate']=='20/1'
    requests=[dict(name='opening',stage='card',frame=30)]
    for c in manifest['clips']:
        for stage,index in [('initial',0),('middle',len(c['mapping'])//2),('final',len(c['mapping'])-1)]:
            requests.append(dict(name=c['name'],stage=stage,frame=c['start_frame']+index))
    requests.append(dict(name='closing',stage='card',frame=manifest['frames']-30))
    assert len(requests)==2+3*len(manifest['clips'])
    expression="select='"+'+'.join(f"eq(n,{r['frame']})" for r in requests)+"'"
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-threads','2','-i',str(video),'-vf',expression,'-fps_mode','vfr',str(dest/'encoded-%03d.png')],check=True)
    for i,r in enumerate(requests,1):
        path=dest/f'encoded-{i:03d}.png';assert path.exists();r.update(path=str(path),sha256=sha(path))
    assert sha(E/(base+'-audit.json'))==manifest['audit_sha256']
    audit=read(E/(base+'-audit.json'));indexed={r['name']:r for r in audit['records']};comparisons=[]
    for c in manifest['clips']:
        r=indexed[c['name']];assert r['sha256']==c['source_sha256']
        identity=dict(r,relative_path=str(PurePosixPath(r['path']).relative_to('/workspace/outputs/fountain')))
        with verified_recording(identity) as source,h5py.File(source) as f:
            ds=f['data/demo_0']
            for request in (q for q in requests if q['name']==c['name']):
                group,index=c['mapping'][request['frame']-c['start_frame']]
                encoded=Image.open(request['path']).convert('RGB')
                for camera,x,size in [('overview',24,512),('cup_detail',568,320),('wrist',920,320)]:
                    native=picture(ds[group+'/'+camera+'_image'],index)
                    if camera=='cup_detail':
                        u,v=r['cameras'][camera]['rim_center_uv'];native=native.crop((round(u-160),round(v-80),round(u+160),round(v+240)))
                    native=native.resize((size,size),Image.Resampling.LANCZOS)
                    error=float(abs(np.asarray(native,dtype=float)-np.asarray(encoded.crop((x,112,x+size,112+size)),dtype=float)).mean())
                    assert error<2.5,(request,camera,error)
                    comparisons.append(dict(name=c['name'],stage=request['stage'],camera=camera,mean_absolute_pixel_error=error))
    for page in range((len(requests)-2+5)//6):
        sheet=Image.new('RGB',(1280,544),'#101820');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(requests[1:-1][page*6:(page+1)*6]):
            x=j%3*426;y=j//3*272;sheet.paste(Image.open(r['path']).resize((426,240),Image.Resampling.LANCZOS),(x,y))
            draw.text((x+6,y+244),r['name']+' / '+r['stage'],font=font(13),fill='white')
        sheet.save(dest/f'contact-{page+1}.jpg',quality=95)
    report=dict(video_sha256=manifest['sha256'],full_decode_passed=True,frames=manifest['frames'],duration_s=float(probe['format']['duration']),
        encoded_review_frames=requests,camera_comparisons=comparisons,maximum_mean_absolute_pixel_error=max(r['mean_absolute_pixel_error'] for r in comparisons),
        all_clips_have_initial_middle_final=True,visual_review='pending',helper_sha256=sha(Path(__file__)))
    (E/(stem+'-video-review.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(full_decode=True,selected_encoded_frames=len(requests),camera_comparisons=len(comparisons),visual_review='pending')))
if __name__=='__main__':main()
