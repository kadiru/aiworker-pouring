"""CPU-only recording integrity and native camera review for this study."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from validate_pour_recording import validate,export_views,decode,h5py
from pour_settled_study import ROOT,PREFIX


def main(mode,development):
    report=json.loads((ROOT/'settled-results.json').read_text())
    if not development:assert report['status']=='complete'
    rows=report['development'] if development else (report['all_attempts'] if mode=='integrity' else report['cases'])
    assert rows
    if mode=='integrity':
        base=Path('/workspace/outputs/fountain');files=[];sidecars=[]
        assert not list(base.glob(f'*/*{PREFIX}*/episode.partial.hdf5')),'Preserve and handle incomplete attempts explicitly'
        for row in rows:
            p=Path(row['path'])/'episode.hdf5';v=validate(p);assert v['valid'];export_views(p)
            files.append(dict(name=row['name'],relative_path=str(p.relative_to(base)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=True,transitions=v['transitions'],
                physical_success=v['physical_success'],max_camera_state_skew_s=v['max_camera_state_skew_s']))
            for name in ('parameters.json','metrics.json','samples.json','policy-decisions.json'):
                q=p.parent/name;sidecars.append(dict(relative_path=str(q.relative_to(base)),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
        out=ROOT/('development-integrity.json' if development else 'recording-integrity.json')
        out.write_text(json.dumps(dict(files=files,sidecar_files=sidecars,valid_recordings=len(files),note='Every completed attempt included, including task failures; recording validity is separate.'),indent=2)+'\n')
        print('Validated',len(files),'HDF5 and',len(sidecars),'sidecars')
    else:
        for page in range((len(rows)+5)//6):
            group=rows[page*6:(page+1)*6];sheet=Image.new('RGB',(1536,444*len(group)),(20,25,32));draw=ImageDraw.Draw(sheet)
            for n,row in enumerate(group):
                with h5py.File(Path(row['path'])/'episode.hdf5','r') as f:
                    d=f['data/demo_0'];st=np.flatnonzero(d['ground_truth/stop_command'][:]);i=int(st[0]) if len(st) else int(np.argmax(d['ground_truth/cup_ml'][:]))
                    draw.text((8,n*444+5),f'{row["name"]} | success={row["surface_goal_zero_spill_success"]} | height={row["cup_surface_height_fraction"]:.4f} | spill={row["final_outside_ml"]:.4f}',fill='white')
                    for col,(ix,cam,label) in enumerate(((i,'task','stop task'),(i,'wrist','stop wrist'),(len(d['actions'])-1,'task','final task'),(len(d['actions'])-1,'wrist','final wrist'))):
                        draw.text((col*384+8,n*444+25),label,fill='white');sheet.paste(Image.fromarray(decode(d['next_obs/'+cam+'_image'][ix],f.attrs['image_encoding'])),(col*384,n*444+48))
            out=ROOT/f'{"development" if development else "camera"}-review-{page+1:02}.jpg';sheet.save(out,quality=94);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('integrity','review'));p.add_argument('--development',action='store_true');a=p.parse_args();main(a.mode,a.development)
