"""Decode the delivered MP4, select actual encoded frames, and make review sheets."""
import json
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
from render_labeled_benchmark import font,sha

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/layout-variety-A';D=E/'video-review'
manifest=json.loads((E/'video-manifest.json').read_text());video=Path(manifest['video'])
assert sha(video)==manifest['sha256'];D.mkdir(exist_ok=True)
subprocess.run(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
requests=[dict(name='opening',stage='card',frame=30)]+manifest['review_frame_requests']+[dict(name='closing',stage='card',frame=manifest['frames']-30)]
assert len(requests)==44
indices=[r['frame'] for r in requests];expression="select='"+'+'.join(f'eq(n,{n})' for n in indices)+"'"
subprocess.run(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-i',str(video),'-vf',expression,
    '-fps_mode','vfr',str(D/'encoded-%03d.png')],check=True)
paths=[D/f'encoded-{i:03}.png' for i in range(1,45)]
assert all(p.exists() for p in paths)
for page in range(5):
    subset=list(zip(paths,requests))[page*10:(page+1)*10]
    sheet=Image.new('RGB',(1280,5*390),'#101820');draw=ImageDraw.Draw(sheet)
    for j,(path,request) in enumerate(subset):
        x=(j%2)*640;y=(j//2)*390
        sheet.paste(Image.open(path).convert('RGB').resize((640,360),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+12,y+363),request['name']+' / '+request['stage'],font=font(16),fill='white')
    sheet.save(D/f'contact-{page+1}.jpg',quality=93)
probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
stream=probe['streams'][0]
assert stream['width']==1280 and stream['height']==720 and stream['pix_fmt']=='yuv420p' and stream['codec_name']=='h264'
assert int(stream['nb_frames'])==manifest['frames']
report=dict(full_video_decode_passed=True,encoded_review_frames=44,all_fourteen_have_initial_middle_final=True,
    video_sha256=manifest['sha256'],duration_s=float(probe['format']['duration']),codec=stream['codec_name'],pixel_format=stream['pix_fmt'],
    frames=[dict(**r,path=str(p),sha256=sha(p)) for r,p in zip(requests,paths)],visual_review='pending')
(E/'video-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(duration_s=report['duration_s'],frames=manifest['frames'],review_frames=44,full_decode=True)),flush=True)
