"""Export a prespecified closed test's real task/wrist camera frames at 20 Hz.

Ubuntu only reads the closed JPEG recording in the capped container. Decoding,
composition and H.264 encoding run on the local computer. Requires Pillow and
ffmpeg/ffprobe locally; never sends a simulator command.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / 'evidence/workspace-A'
READER = '''
import hashlib,json,struct,sys
from pathlib import Path
from validate_pour_recording import h5py
name,sha=sys.argv[1:];base=Path('/workspace/outputs/fountain')
folders=list(base.glob('*/*-'+name+'-*'));assert len(folders)==1
folder=folders[0];p=folder/'episode.hdf5';m=json.loads((folder/'metrics.json').read_text());s=json.loads((folder/'parameters.json').read_text())
assert s['name']==name and m['policy']['checkpoint_sha256']==sha and not m['policy']['teacher_fallback']
with h5py.File(p,'r') as f:
    d=f['data/demo_0'];assert f.attrs['status']=='complete' and f.attrs['image_encoding']=='jpeg95' and abs(float(f.attrs['control_dt'])-.05)<1e-8
    n=len(d['actions']);b=m['bottle_result'];header=dict(name=name,frames=n,camera_fps=20,source_hdf5=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checkpoint_sha256=sha,expected_stop=s['expected_stop'],actual_stop=b['stop_reason'],stop_time=b['stop_time'],strict_success=m['surface_task_success_zero_spill'],final_rim_height=b['cup_fill_fraction'],source_remaining_nominal_ml=b['bottle_ml'],spill=m['spill'])
    out=sys.stdout.buffer;out.write((json.dumps(header)+'\\n').encode())
    for i in range(n):
        a=d['next_obs/task_image'][i].tobytes();w=d['next_obs/wrist_image'][i].tobytes();phase=d['ground_truth/phase'].asstr()[i].encode()
        out.write(struct.pack('>IIdH',len(a),len(w),float(d['next_time'][i]),len(phase))+phase+a+w)
    out.flush()
'''


def read_exact(stream, n):
    chunk = stream.read(n)
    if len(chunk) != n:
        raise EOFError('Closed recording stream ended before the declared frame count')
    return chunk


def main(name):
    plan = json.loads((EVIDENCE/'frozen-plan.json').read_text())
    chosen = {next(x['name'] for x in plan['test'] if x['expected_stop']==g): label
              for g,label in [('cup_near_full','full'),('bottle_empty','empty')]}
    assert name in chosen, 'Only the prespecified full/empty example cases are exported'
    frozen = json.loads((EVIDENCE/'final-selection.json').read_text())
    before = json.loads((EVIDENCE/'runtime-before-tests.json').read_text())
    out = EVIDENCE/f'fresh-{chosen[name]}.mp4'; partial=out.with_suffix('.partial.mp4')
    assert not out.exists(), 'A completed export already exists; preserve it'
    argv=['ssh','kfuUbuntu','docker','exec','-i','-w','/workspace/src',before['container'],
          'nice','-n','10','/isaac-sim/python.sh','-',name,frozen['selected_checkpoint_sha256']]
    with subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as src:
        src.stdin.write(READER.encode());src.stdin.close()
        meta=json.loads(src.stdout.readline());assert meta['name']==name and meta['frames']>20
        font_path=Path('/System/Library/Fonts/Helvetica.ttc')
        font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
        small=ImageFont.truetype(str(font_path),15) if font_path.exists() else font
        encoder=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                 '-video_size','768x460','-framerate','20','-i','pipe:0','-an','-c:v','libx264',
                 '-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
        with subprocess.Popen(encoder,stdin=subprocess.PIPE) as enc:
            for _ in range(meta['frames']):
                na,nw,t,np=struct.unpack('>IIdH',read_exact(src.stdout,18));assert 0<na<2**21 and 0<nw<2**21 and np<1024
                phase=read_exact(src.stdout,np).decode().removeprefix('Visual guard: ')
                canvas=Image.new('RGB',(768,460),'#162331');draw=ImageDraw.Draw(canvas)
                draw.text((10,8),'Task camera',font=font,fill='white');draw.text((394,8),'Wrist camera',font=font,fill='white')
                for x,n in ((0,na),(384,nw)):
                    with Image.open(io.BytesIO(read_exact(src.stdout,n))) as frame:
                        assert frame.size==(384,384);canvas.paste(frame.convert('RGB'),(x,34))
                draw.text((10,429),f'{name}   |   {t:05.2f} s   |   {phase}',font=small,fill='white')
                enc.stdin.write(canvas.tobytes())
            enc.stdin.close();assert enc.wait()==0
        assert src.stdout.read(1)==b'' and src.wait()==0
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate,width,height,duration','-of','json',str(partial)]))['streams'][0]
    assert int(info['nb_read_frames'])==meta['frames'] and info['r_frame_rate']=='20/1'
    assert abs(float(info['duration'])-meta['frames']/20)<.002
    partial.replace(out);meta.update(video=out.name,video_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        video_stream=info,provenance='Every next_obs task/wrist JPEG in native order; recorded phase and simulation time overlaid. No generated frames.')
    (EVIDENCE/f'fresh-{chosen[name]}-video-review.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');main(p.parse_args().case)
