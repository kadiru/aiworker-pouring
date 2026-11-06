"""Search cup masks on consumed references/development using CPU only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from itertools import product


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(a.candidate_dir),'/workspace/src']
    import numpy as np,torch
    from validate_pour_recording import h5py,decode
    from pour_guard_vision import make_bank,locate,project,plane_point
    root=Path('/workspace/learning/A-wide-20261105-v1');output=a.candidate_dir/'cup-candidates.json'
    assert not output.exists() and not (root/'final-selection.json').exists();torch.set_num_threads(2);start=time.monotonic()
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False);accepted=set(json.loads((root/'reference-acceptance.json').read_text())['accepted']);seq=[]
    paths=sorted(Path('/workspace/outputs/fountain').glob('*/*T30_A_demo_*/episode.hdf5'))+sorted(Path('/workspace/outputs/fountain').glob('*/*XW30_A_dev_*_r0*/episode.hdf5'))
    for path in paths:
        with h5py.File(path) as f:
            s=json.loads(f.attrs['spec'])
            if s.get('protocol')!=17 or (s['name'].startswith('T30') and s['name'] not in accepted):continue
            assert s['study_split']!='test' and f.attrs['status']=='complete'
            d=f['data/demo_0'];image=decode(d['obs/task_image'][0],'jpeg95').copy();xyz=np.asarray(s['cup_world'])
            uv=project(xyz+[0,0,.038],saved['calibration']);x,y=np.rint(uv).astype(int);negative=image.copy()
            negative[max(0,y-18):min(384,y+22),max(0,x-44):min(384,x+44)]=180
            seq.append(dict(name=s['name'],split=s['study_split'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image=image,negative=negative,xyz=xyz,uv=uv))
    train=[e for e in seq if e['split']=='train'];assert len(train)==24
    screen=[e for e in seq if e['split']=='validation' or e['name']=='XW30_A_dev_04_r0'];assert len(screen)==9
    results=[]
    for radius,mask,clip,aggregation in product((32.,48.),(None,(10,2,70,24),(16,4,64,22),(10,0,70,16)),(None,.05,.1),('max','top3')):
        bank=make_bank([e['image'] for e in train],[e['uv'] for e in train],'cup',device='cpu',aggregation=aggregation,absolute_gradients=True,anchor_radius_px=radius,cup_mask_box=mask,gradient_clip=clip)
        for key in ('bounds','initial_region_uv','initial_workspace_xy_m'):bank[key]=saved['cup_bank'][key]
        def evaluate(items):
            rows=[]
            for e in items:
                uv,score=locate(e['image'],bank,device='cpu');_,negative=locate(e['negative'],bank,device='cpu')
                xy=plane_point(uv,1.078,saved['calibration'])[:2]
                rows.append(dict(name=e['name'],split=e['split'],error_mm=float(np.linalg.norm(xy-e['xyz'][:2])*1000),score=score,synthetic_negative_score=negative))
            return rows
        rows=evaluate(screen);margin=min(r['score'] for r in rows)-max(r['synthetic_negative_score'] for r in rows)
        if max(r['error_mm'] for r in rows)<2 and margin>.015:rows=evaluate(seq)
        minimum=min(r['score'] for r in rows);maximum=max(r['synthetic_negative_score'] for r in rows)
        r=dict(radius=radius,cup_mask_box=mask,gradient_clip=clip,aggregation=aggregation,rows=rows,max_error_mm=max(r['error_mm'] for r in rows),margin=minimum-maximum,threshold=(minimum+maximum)/2,complete=len(rows)==len(seq))
        results.append(r);print('CUP_CANDIDATE',radius,mask,clip,aggregation,'error',r['max_error_mm'],'margin',r['margin'],'complete',r['complete'],flush=True)
        output.write_text(json.dumps(dict(status='running',candidates=results),indent=2)+'\n')
    valid=[r for r in results if r['complete'] and r['max_error_mm']<2 and r['margin']>.015]
    report=dict(status='passed' if valid else 'failed',device='cpu',cuda_initialized=torch.cuda.is_initialized(),elapsed_s=time.monotonic()-start,source_sha256=hashlib.sha256((a.candidate_dir/'pour_guard_vision.py').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),training_names=[e['name'] for e in train],consumed_records=[{k:e[k] for k in ('name','split','path','sha256')} for e in seq],candidates=results,selected=min(valid,key=lambda r:(r['max_error_mm'],-r['margin'])) if valid else None,scope='Only 24 training reference images supply templates. Reference validation and already consumed development images select masks. No test data or deployed/checkpoint mutation. Erasure negatives are synthetic.')
    assert not report['cuda_initialized'];output.write_text(json.dumps(report,indent=2)+'\n');print('CUP_SEARCH_DONE',report['status'],flush=True)


if __name__=='__main__':main()
