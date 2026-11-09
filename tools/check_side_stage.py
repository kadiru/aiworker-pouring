"""Read-only integrity, label and camera audit of the new native study."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from collections import Counter
from PIL import Image,ImageDraw
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode,validate
from pour_perception_geometry import settled_label
from pour_guard_vision import project
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    plan=read(ROOT/'plan.json');candidate=read(ROOT/'candidate.json')
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    assert all(sha(Path(p))==v for p,v in candidate['protected_parent_files'].items())
    expected={s['name']:s for part in ('diagnostic','impact','impact_followup','development','test') for s in plan.get(part,[])}
    result=[];seen=set();out=ROOT/'review';out.mkdir(exist_ok=True)
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*SR_A_*/parameters.json')):
        spec=read(path);name=spec['name'];assert name in expected and name not in seen
        seen.add(name);assert all(spec[k]==v for k,v in expected[name].items())
        h5=path.parent/'episode.hdf5'
        if not h5.exists() or not (path.parent/'metrics.json').exists():continue
        validation=validate(h5);samples=read(path.parent/'samples.json')
        r=dict(name=name,split=spec['study_split'],path=str(h5),sha256=sha(h5),bytes=h5.stat().st_size,
            transitions=validation['transitions'],max_camera_state_skew_s=validation['max_camera_state_skew_s'],
            sidecars={p.name:sha(p) for p in path.parent.glob('*.json')})
        trajectory=path.parent/'trajectory.npz'
        if trajectory.exists():r['trajectory_sha256']=sha(trajectory)
        if spec.get('calibration_only'):
            r['label']=settled_label(samples)
            if spec.get('record_surface_diagnostics'):
                r['column_label']=settled_label(samples,field='cup_column_height_fraction')
                trajectory=path.parent/'trajectory.npz'
                assert trajectory.exists()
                r['trajectory_sha256']=sha(trajectory)
        with h5py.File(h5) as f:
            d=f['data/demo_0'];r['cameras']={};cameras=('task','wrist','cup_detail')+(('overview',) if spec.get('record_overview') else ());canvas=Image.new('RGB',(384*len(cameras),408),(245,245,245));draw=ImageDraw.Draw(canvas)
            assert np.allclose(f['calibration/task/intrinsics'][:],saved['calibration']['K'],rtol=0,atol=1e-3)
            assert np.allclose(d['obs/task_world_T_camera'][:],saved['calibration']['T'],rtol=0,atol=1e-6)
            r['legacy_task_camera_unchanged']=True
            for i,camera in enumerate(cameras):
                a=decode(d['next_obs/'+camera+'_image'][-1],'jpeg95')
                cal=dict(K=f['calibration/'+camera+'/intrinsics'][:].tolist(),T=d['next_obs/'+camera+'_world_T_camera'][-1].tolist(),table_z=1.04)
                r['cameras'][camera]=dict(calibration=cal,resolution=[a.shape[1],a.shape[0]])
                if camera=='cup_detail':
                    size=spec.get('cup_detail_resolution',768);focal=spec.get('cup_detail_focal_m',.045)
                    assert a.shape==(size,size,3)
                    assert abs(cal['K'][0][0]-size*focal*1000/20.955)<1e-3
                Image.fromarray(a).save(out/(name+'-'+camera+'.jpg'))
                canvas.paste(Image.fromarray(a).resize((384,384)),(384*i,24));draw.text((384*i+8,7),name+' | '+camera,fill=(20,20,20))
                if camera!='wrist':
                    from pour_vessel_geometry import cup_geometry
                    g=cup_geometry(spec);center=np.asarray(spec['cup_world'])
                    low=project(center+[0,0,g['bottom_z_m']],cal);high=project(center+[0,0,g['rim_z_m']],cal)
                    r['cameras'][camera]['projected_inner_height_px']=float(np.linalg.norm(high-low))
                    r['cameras'][camera]['rim_center_uv']=high.tolist()
            canvas.save(out/(name+'-views.jpg'))
        result.append(r)
    report=dict(records=result,protected_parent_files_unchanged=True,unstarted=sorted(set(expected)-seen),
        started_incomplete=sorted(seen-{r['name'] for r in result}),
        scope='Native recording integrity and camera audit. Authored stationary/held fixtures are calibration only; autonomous outcomes are scored separately.',helper_sha256=sha(Path(__file__)))
    (ROOT/'recording-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=len(result),accepted_labels=sum(r.get('label',{}).get('accepted',False) for r in result),
        closed_by_split=dict(Counter(r['split'] for r in result)),
        rejected_labels=[r['name'] for r in result if 'label' in r and not r['label']['accepted']],
        latest_cases=[dict(name=r['name'],label=r.get('label'),detail_inner_height_px=r['cameras']['cup_detail']['projected_inner_height_px']) for r in result[-3:]],incomplete=report['started_incomplete'])))

if __name__=='__main__':main()
