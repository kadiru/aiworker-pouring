"""Export synchronized recorded RGB frames for CPU video encoding, no simulator."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from validate_pour_recording import decode
import h5py


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.episode,'r') as f:
        d=f['data/demo_0'];codec=f.attrs.get('image_encoding','raw')
        spec=json.loads(f.attrs['spec']);kind=f.attrs['evidence_kind']
        for frame,i in enumerate(range(0,len(d['actions']),2)):
            image=Image.new('RGB',(768,432),(18,26,32));draw=ImageDraw.Draw(image)
            for j,c in enumerate(('task','wrist')):
                image.paste(Image.fromarray(decode(d['next_obs/'+c+'_image'][i],codec)),(j*384,32))
            draw.text((8,8),spec['name']+' | '+kind+' | simulated cameras',fill='white')
            if 'expert_supervision' in d:
                draw.text((8,20),'CONTROL: '+('SCRIPTED EXPERT RECOVERY' if d['expert_supervision'][i] else 'LEARNED POLICY PREFIX'),fill='white')
            t=float(d['next_time'][i]);cup=float(d['ground_truth/cup_ml'][i]);spill=float(d['ground_truth/spilled_ml'][i])
            reason=d['ground_truth/stop_reason'][i].decode() or 'no stop decision'
            draw.text((8,418),f'{t:.2f} sim s | cup {cup:.2f} nominal mL | spilled {spill:.3f} mL | {reason}',fill='white')
            image.save(args.output/f'{frame:05}.jpg',quality=94)
        (args.output/'video.json').write_text(json.dumps(dict(fps=10,episode=str(args.episode),evidence_kind=kind),indent=2)+'\n')
    print(args.output)
