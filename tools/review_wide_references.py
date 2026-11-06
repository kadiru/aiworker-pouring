"""Native reference contact sheets for manual inspection, never fake tests."""
import argparse,io,json,hashlib
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw
from render_labeled_benchmark import font

PROJECT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(exist_ok=False)
    manifest=json.loads(a.manifest.read_text());entries=[]
    for item in manifest['files']:
        if not item['name'].startswith('T30_A_demo_'):continue
        p=PROJECT/'data/material-A-wide'/item['relative_path'];assert sha(p)==item['sha256']
        entries.append((item,p))
    reviewed=[]
    for page in range((len(entries)+2)//3):
        group=entries[page*3:(page+1)*3];sheet=Image.new('RGB',(1536,448*len(group)),'#101820');draw=ImageDraw.Draw(sheet)
        for n,(item,p) in enumerate(group):
            with h5py.File(p) as f:
                spec=json.loads(f.attrs['spec']);metrics=json.loads(f.attrs['metrics']);d=f['data/demo_0'];times=d['next_time'][:]
                grasp=np.flatnonzero(d['ground_truth/grasp_verified'][:]);stop=np.flatnonzero(d['ground_truth/stop_command'][:])
                grip_index=int(grasp[0]) if len(grasp) else int(np.argmax(d['ground_truth/bottle_pose'][:,2]))
                stop_index=int(stop[0]) if len(stop) else len(times)-1
                accepted=metrics['grasp_verified'] and metrics['zero_spill'] and not metrics['gates']
                title=f"{spec['name']} | SCRIPTED REFERENCE | {'accepted' if accepted else 'rejected'} | loss {metrics['spill']['final_outside_ml']:.6f} nominal mL"
                draw.text((8,n*448+5),title,font=font(15,True),fill='white')
                views=[(0,'task','initial'),(grip_index,'wrist','grip' if len(grasp) else 'no verified grip'),
                       (stop_index,'wrist','stop'),(len(times)-1,'task','final')]
                samples=[]
                for col,(i,cam,label) in enumerate(views):
                    image=Image.open(io.BytesIO(d['next_obs/'+cam+'_image'][i].tobytes())).convert('RGB')
                    assert image.size==(384,384)
                    sheet.paste(image,(col*384,n*448+58))
                    draw.text((col*384+8,n*448+30),f'{label} / {cam} / {times[i]:.2f}s',font=font(15),fill='#B3C0CC')
                    samples.append(dict(index=i,camera=cam,t=float(times[i])))
                reviewed.append(dict(name=spec['name'],source_sha256=item['sha256'],page=page+1,samples=samples,manual_review='pending'))
        sheet.save(a.output/f'page-{page+1:02}.jpg',quality=95)
    (a.output/'manifest.json').write_text(json.dumps(dict(source_manifest_sha256=sha(a.manifest),references=reviewed),indent=2)+'\n')
    print(json.dumps(dict(pages=(len(entries)+2)//3,references=len(entries),manual_review='pending')))


if __name__=='__main__':main()
