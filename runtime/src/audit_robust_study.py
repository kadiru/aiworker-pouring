"""CPU recording integrity and five native camera sheets for all30 cases."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from validate_pour_recording import validate,export_views,decode,h5py
from pour_robust_study import ROOT,PREFIX


def integrity():
    report=json.loads((ROOT/'robust-results.json').read_text());assert report['status']=='complete'
    base=Path('/workspace/outputs/fountain');files=[];sidecars=[]
    assert not list(base.glob(f'*/*{PREFIX}*/episode.partial.hdf5')),'An incomplete attempt needs explicit handling'
    for row in report['cases']:
        p=Path(row['path'])/'episode.hdf5';r=validate(p);assert r['valid'];export_views(p)
        files.append(dict(relative_path=str(p.relative_to(base)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            valid=True,transitions=r['transitions'],duration_sim_s=r['duration_sim_s'],physical_success=r['physical_success'],max_camera_state_skew_s=r['max_camera_state_skew_s']))
        for name in ('parameters.json','metrics.json','samples.json','policy-decisions.json'):
            q=p.parent/name;sidecars.append(dict(relative_path=str(q.relative_to(base)),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    assert len(files)==30 and len(sidecars)==120
    (ROOT/'recording-integrity.json').write_text(json.dumps(dict(files=files,sidecar_files=sidecars,valid_recordings=30,
        note='Technical recording validity is separate from task success; all failed native attempts are included.'),indent=2)+'\n')
    print('Validated30 HDF5 and120sidecars')


def review():
    report=json.loads((ROOT/'robust-results.json').read_text());assert report['status']=='complete';rows=report['cases']
    for page in range(5):
        sheet=Image.new('RGB',(1536,444*6),(20,25,32));draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(rows[page*6:(page+1)*6]):
            with h5py.File(Path(row['path'])/'episode.hdf5','r') as f:
                d=f['data/demo_0'];st=np.flatnonzero(d['ground_truth/stop_command'][:]);i=int(st[0]) if len(st) else int(np.argmax(d['ground_truth/cup_ml'][:]))
                count=len(d['actions']);draw.text((8,n*444+5),f'{row["name"]} | success={row["surface_goal_zero_spill_success"]} | legacy={row["legacy_inventory_zero_spill_success"]} | level={row["cup_surface_height_fraction"]:.3f} | spill={row["final_outside_ml"]:.3f}',fill='white')
                for col,(ix,cam,label) in enumerate(((i,'task','stop task'),(i,'wrist','stop wrist'),(count-1,'task','final task'),(count-1,'wrist','final wrist'))):
                    draw.text((col*384+8,n*444+25),label,fill='white');sheet.paste(Image.fromarray(decode(d['next_obs/'+cam+'_image'][ix],f.attrs['image_encoding'])),(col*384,n*444+48))
        p=ROOT/f'camera-review-{page+1:02}.jpg';sheet.save(p,quality=94);print(p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('integrity','review'));globals()[p.parse_args().mode]()
