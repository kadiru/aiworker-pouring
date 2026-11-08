"""Calibrate an alternative RGB rim matcher on train/validation only."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import project,plane_point
from pour_cup_rim import locate_rim
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=read(ROOT/'plan.json');audit=read(ROOT/'recording-audit.json');indexed={r['name']:r for r in audit['records']}
    xy=np.array([s['cup_world'][:2] for s in plan['collect'] if s['study_split']=='train']);seq=[]
    for s in plan['collect']:
        r=indexed[s['name']];assert sha(Path(r['path']))==r['sha256']
        with h5py.File(r['path']) as f:
            d=f['data/demo_0'];image=decode(d['obs/cup_detail_image'][11],'jpeg95')
            cal=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[1024,1024])
        seq.append((s,image,cal))
    results=[];start=time.perf_counter()
    for depth in (0.,.003,.005,.007):
        bank=dict(kind='calibrated_cup_rim',calibration=seq[0][2],xy_min=(xy.min(0)-.025).tolist(),xy_max=(xy.max(0)+.025).tolist(),radial_offsets_m=[.0018,.0024,.003],rim_feature_depth_m=depth,gradient_clip=.15,min_red_width_fraction=.5)
        rows=[]
        for s,image,cal in seq:
            actual=dict(bank,geometry=s['declared_cup_geometry']);uv,score=locate_rim(image,actual)
            point=plane_point(uv,1.04+s['declared_cup_geometry']['rim_z_m'],cal)[:2]
            truth=project(np.array(s['cup_world'])+[0,0,s['declared_cup_geometry']['rim_z_m']],cal);x,y=np.rint(truth).astype(int)
            negative=image.copy();negative[max(0,y-54):min(1024,y+94),max(0,x-120):min(1024,x+120)]=180
            _,absent=locate_rim(negative,actual)
            row=dict(name=s['name'],split=s['study_split'],error_mm=float(np.linalg.norm(point-s['cup_world'][:2])*1000),score=score,synthetic_absent_score=absent,localized_xy=point.tolist())
            rows.append(row)
        minimum=min(r['score'] for r in rows);maximum=max(r['synthetic_absent_score'] for r in rows)
        result=dict(bank=bank,rows=rows,max_error_mm=max(r['error_mm'] for r in rows),margin=minimum-maximum,threshold=(minimum+maximum)/2)
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('bank','rows')}),flush=True)
    report=dict(candidates=results,total_seconds=time.perf_counter()-start,helper_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'staged/pour_cup_rim.py'),scope='Train/validation calibration only; declared dimensions and fixed camera calibration. No reserved images or runtime object poses.')
    (ROOT/'rim-localization-probe.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
