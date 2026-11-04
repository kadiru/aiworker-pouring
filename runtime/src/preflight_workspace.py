"""CPU-only reachability and camera coverage; no simulation or robot commands."""
import hashlib
import json
import os
import sys
from pathlib import Path
bindings=list(Path('/isaac-sim/extscache').glob('omni.usd.libs-*'))
assert len(bindings)==1, 'Inspect the installed USD binding version before preflight'
if not os.environ.get('AIWORKER_USD_BOOTSTRAPPED'):
    env=dict(os.environ,AIWORKER_USD_BOOTSTRAPPED='1')
    env['LD_LIBRARY_PATH']=str(bindings[0]/'bin')+':'+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
import numpy as np
import torch
sys.path.append(str(bindings[0]))
from pxr import Usd,UsdPhysics
from robot_pour import build_path,ROOT as ROBOT_ROOT
from pour_guard_vision import project
from pour_workspace_study import ROOT,PARENT,PARENT_SHA,plan


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256((PARENT/'best.pt').read_bytes()).hexdigest()==PARENT_SHA
    saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
    stage=Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/Robot','Xform').GetReferences().AddReference('/workspace/assets/FFW_BG2_without_mimic.usd')
    assert stage.GetPrimAtPath(ROBOT_ROOT+'/joints/arm_r_joint1')
    p=plan(); rows=[]
    for spec in p['demonstrations']+p['development']+p['test']:
        row=dict(name=spec['name'])
        try:
            paths=build_path(stage,spec);row['reachable']=all(np.isfinite(a).all() for a in paths)
            row['poses_solved']=sum(len(a) for a in paths)
        except ValueError as e:row.update(reachable=False,error=str(e))
        cx,cy,z=spec['cup_world'];sx,sy=spec['source_xy']
        row['source_uv']=project([cx-sy,cy+sx,z+.1352],saved['calibration']).tolist()
        row['cup_uv']=project([cx,cy,z+.038],saved['calibration']).tolist()
        # Entire matching patches must fit the 384x384 task camera.
        u,v=row['source_uv'];a,b=row['cup_uv']
        row['camera_patches_visible']=24<u<360 and 12<v<332 and 40<a<344 and 14<b<370
        rows.append(row)
    report=dict(kind='CPU kinematics and calibrated projection; not collision/contact or successful-grasp evidence',
        planned=len(rows),reachable=sum(x['reachable'] for x in rows),camera_covered=sum(x['camera_patches_visible'] for x in rows),cases=rows)
    (ROOT/'workspace-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    assert all(x['reachable'] and x['camera_patches_visible'] for x in rows)


if __name__=='__main__':main()
