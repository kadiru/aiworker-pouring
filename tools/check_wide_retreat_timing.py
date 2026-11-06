"""CPU replay of terminal timing on closed development joint histories.

Checks planned joint increments, not native dynamics or collision clearance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(a.candidate_dir),'/workspace/src']
    import torch
    import numpy as np
    from pour_guard_actor import gentle_terminal_profile
    from pour_stop_sequence import TerminalPrimitive
    from validate_pour_recording import h5py
    torch.set_num_threads(2)
    root=Path('/workspace/learning/A-wide-20261105-v1');saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    output=a.candidate_dir/'retreat-timing.json';assert not output.exists();cases=[]
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*XW30_A_dev_*_r1*/episode.hdf5')):
        decisions=json.loads((path.parent/'policy-decisions.json').read_text())
        with h5py.File(path) as f:
            spec=json.loads(f.attrs['spec']);assert spec['study_split']=='development' and f.attrs['status']=='complete'
            q=f['data/demo_0/obs/joint_pos'][:]
        first=next(i for i,r in enumerate(decisions) if r['mode']=='Visual grasp: approach')
        grasp=next(r['t'] for r in decisions if r['grasp_stage']=='lift')
        lift=next(r['t'] for r in decisions if r['grasp_stage']=='ready')
        stop=next(i for i,r in enumerate(decisions) if r['stop_latched'])
        history=[(decisions[i]['t'],q[i,:7]) for i in range(first,stop+1)]
        old=dict(saved['primitive'],grasp_mark_s=grasp,lift_mark_s=lift,return_s=4.3)
        new=gentle_terminal_profile(old,dict(place_s=1.5,release_s=.6,retreat_s=2.))
        versions={}
        for label,profile in [('old',old),('candidate',new)]:
            actor=TerminalPrimitive(history,0.,profile);start=sum(profile[k] for k in ('return_s','place_s','release_s'))
            times=np.r_[np.arange(0.,profile['retreat_s'],.05),profile['retreat_s']]
            actions=np.stack([actor.act(start+float(t))[0] for t in times])
            assert np.max(abs(actions[:,7:]))==0.
            versions[label]=dict(retreat_s=profile['retreat_s'],place_s=profile['place_s'],release_s=profile['release_s'],
                peak_planned_joint_step_rad=float(np.max(abs(np.diff(actions[:,:7],axis=0)))))
        native=[r for r in decisions if r['mode']=='Retreating']
        cases.append(dict(name=spec['name'],recording_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            versions=versions,native_retreat_frames=len(native),native_clamped_retreat_frames=sum(r['delta_clamped'] for r in native)))
    assert cases and not torch.cuda.is_initialized()
    result=dict(device='cpu',cuda_initialized=False,cases=cases,
        candidate_max_planned_joint_step_rad=max(r['versions']['candidate']['peak_planned_joint_step_rad'] for r in cases),
        scope='Closed consumed-development measured joint paths, same paths with slower timing. This does not simulate tracking, contacts or fluid; native regression remains required.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert result['candidate_max_planned_joint_step_rad']<.10,'Retreat commands still approach the .12-radian step limit'


if __name__=='__main__':main()
