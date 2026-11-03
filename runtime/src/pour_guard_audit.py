"""CPU-only model provenance, native recording integrity and final-six review."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from validate_pour_recording import validate,export_views,decode,h5py
from pour_guard_study import ROOT


def provenance():
    p=ROOT/'best.pt';b=torch.load(p,map_location='cpu',weights_only=False)
    keys=('model_version','step','cup_bias','cup_validation','thresholds','method','inputs','training_device','training_frames',
        'localization_validation','no_privileged_runtime_inputs','parent_checkpoint_sha256','scene_plan_sha256','data_manifest_sha256','sources',
        'training_sources','controller_revision_note','full_calibration')
    r={k:b[k] for k in keys if k in b};r['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    r['surface_geometry']={k:v for k,v in b['surface_geometry'].items() if k!='coef'}
    r['pour_primitive']={k:v for k,v in b['pour_primitive'].items() if k!='weights'}
    r['cup_parameter_count']=sum(v.numel() for v in b['cup_model'].values())
    r['source_detector_frozen']=True;r['cuda_initialized_by_audit']=torch.cuda.is_initialized()
    (ROOT/'model-summary.json').write_text(json.dumps(r,indent=2)+'\n');print('model',r['checkpoint_sha256'],'CUDA audit initialized',torch.cuda.is_initialized())


def integrity():
    base=Path('/workspace/outputs/fountain');files=[];sidecars=[];partial=[]
    for p in sorted(base.glob('*/*G_A_*/episode.hdf5')):
        r=validate(p);export_views(p)
        assert r['valid'],str(p)
        files.append(dict(relative_path=str(p.relative_to(base)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            valid=r['valid'],transitions=r['transitions'],duration_sim_s=r['duration_sim_s'],physical_success=r['physical_success'],
            max_camera_state_skew_s=r['max_camera_state_skew_s']))
        for n in ('metrics.json','parameters.json','samples.json','policy-decisions.json'):
            q=p.parent/n;sidecars.append(dict(relative_path=str(q.relative_to(base)),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    for p in sorted(base.glob('*/*G_A_*/episode.partial.hdf5')):
        with h5py.File(p,'r') as f:status=f.attrs['status'];reason=f.attrs.get('abort_reason',f.attrs.get('error',''))
        assert status in ('aborted','incomplete') and reason,'Unexpected active or unexplained partial recording'
        partial.append(dict(path=str(p),relative_path=str(p.relative_to(base)),bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status=status,reason=reason,excluded=True))
    result=dict(files=files,sidecar_files=sidecars,valid_recordings=len(files),partial_aborted_excluded=partial,
        note='Integrity checks do not imply task success. Known aborted development is retained; reserved tests require complete files.')
    (ROOT/'recording-integrity.json').write_text(json.dumps(result,indent=2)+'\n');print('valid',len(files),'partial retained',len(partial))


def review():
    r=json.loads((ROOT/'guard-results.json').read_text());assert r['test_status']=='complete'
    rows=sorted(r['groups']['test']['rows'],key=lambda x:x['name']);sheet=Image.new('RGB',(1536,444*6),(19,24,29));draw=ImageDraw.Draw(sheet)
    for n,row in enumerate(rows):
        with h5py.File(Path(row['path'])/'episode.hdf5','r') as f:
            d=f['data/demo_0'];st=np.flatnonzero(d['ground_truth/stop_command'][:]);i=int(st[0]) if len(st) else int(np.argmax(d['ground_truth/cup_ml'][:]))
            count=len(d['actions']);draw.text((8,n*444+5),f'{row["name"]} | success={row["surface_goal_zero_spill_success"]} | grasp attempts={row["grasp_attempts"]} | pauses={row["pause_count"]} | surface={row["cup_surface_height_fraction"]:.3f} | outside={row["final_outside_ml"]:.3f}',fill='white')
            for col,(ix,cam,label) in enumerate(((i,'task','stop task'),(i,'wrist','stop wrist'),(count-1,'task','final task'),(count-1,'wrist','final wrist'))):
                draw.text((col*384+8,n*444+25),label,fill='white');sheet.paste(Image.fromarray(decode(d['next_obs/'+cam+'_image'][ix],f.attrs['image_encoding'])),(col*384,n*444+48))
    sheet.save(ROOT/'test-contact-sheet.jpg',quality=94);print(ROOT/'test-contact-sheet.jpg')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('provenance','integrity','review'));args=p.parse_args();globals()[args.mode]()
