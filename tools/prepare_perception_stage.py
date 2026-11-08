"""Native preparation; archived parent checkpoints/results are read-only."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
PARENT=Path('/workspace/learning/A-size-20261108-v1')
FINISH=Path('/workspace/learning/A-finish-20261106-v1')
PARENT_SHA='a38f447160d8f9de664494f6d685e628daa9c0d647ecf922d5223b7d892bdeaf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def main():
    sys.path.insert(0,str(ROOT/'staged'))
    from pour_perception_geometry import calibration_seed_points
    from pour_vessel_geometry import cup_geometry
    from bottle_pour import vessel_mask
    from pour_vessel_geometry import cup_profile
    assert not (ROOT/'candidate.json').exists()
    assert sha(PARENT/'best.pt')==PARENT_SHA
    seed=FINISH/'calibration-particle-seed.npz'
    assert sha(seed)=='1c21c07fe9659e8688cb3bc0365c3c96e5a0efd0dcb57ce60233dea66e7680a8'
    with np.load(seed) as f:points=f['cup_local_points'].copy()
    base=read(FINISH/'frozen-plan.json')['development'][0]
    plan=dict(version=1,pilot=[],collect=[],development=[],test=[],
        scope='Static native perception calibration and separate autonomous development/test. Static authored held poses never count as autonomous grasps.',
        training_label='Median final 1 s, constant inventory, IQR <=1 pp, half-window drift <=0.5 pp, no lost fluid.',
        strict_task_metric='Unchanged final-sample >=90% rim or source empty; correct stop, grasp, return and zero loss.',
        seed_sha256=sha(seed))
    checks=[]
    def scene(name,h,d,cut,offset,held,split):
        s=dict(copy.deepcopy(base),name=name,study_split=split,policy_evaluation=False,
            calibration_only=True,calibration_version=4,calibration_initial_height=cut,
            calibration_particle_seed=str(seed),calibration_particle_seed_sha256=sha(seed),
            cup_height_scale=h,cup_diameter_scale=d,initial_ml=20.,max_seconds=3.5,
            record_cup_detail=True,record_overview=False,record_particle_diagnostics=True)
        s.pop('checkpoint',None)
        s['cup_world']=(np.array(base['cup_world'])+[*offset,0.]).tolist()
        if held:s['calibration_hold_angle_deg']=80.
        s['declared_cup_geometry']=cup_geometry(s)
        selected=calibration_seed_points(s,points)
        assert vessel_mask(selected,cup_profile(s)).all()
        s['cup_initial_ml']=len(selected)*s['spacing']**3*1e6
        checks.append(dict(name=name,particles=len(selected),positions_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            seed_cut=cut,initial_max_z=float(selected[:,2].max()),cup_nominal_ml=s['cup_initial_ml']))
        return s
    for i,(cut,offset,held) in enumerate(((.95,(-.015,-.015),False),(.95,(.015,.015),True),(.5,(0.,0.),True))):
        plan['pilot'].append(scene(f'PV_A_pilot_{i:02d}',1.,1.,cut,offset,held,'pilot'))
    sizes=[(h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1)]
    for i,(h,d) in enumerate(sizes):
        for split,cuts,offsets in (
            ('train',(.35,.70,.99),((-.012,-.01),(.012,0.),(0.,.012))),
            ('validation',(.55,.94),((-.006,.006),(.006,-.006))),
            ('perception_test',(.87,),((.003,.003),))):
            for j,(cut,offset) in enumerate(zip(cuts,offsets)):
                plan['collect'].append(scene(f'PV_A_{split}_{i:02d}_{j}',h,d,cut,offset,(i+j)%2==0,split))
    # These jittered autonomous scenes are new realizations, not old test reruns.
    rng=np.random.default_rng(2601108)
    old=read(FINISH/'frozen-plan.json')['test']
    for part,indices in (('development',(13,15,19,23,0,29)),('test',(2,5,8,11,18,25))):
        for j,index in enumerate(indices):
            s=copy.deepcopy(old[index]);s.update(name=f'PV_A_{part}_{j:02d}',study_split=part,
                checkpoint=str(ROOT/'best.pt'),record_cup_detail=True,record_overview=False,
                parent_layout_family=s['name'],development_round=0)
            s['cup_world']=(np.asarray(s['cup_world'])+[*rng.uniform(-.003,.003,2),0.]).tolist()
            s['source_xy']=(np.asarray(s['source_xy'])+rng.uniform(-.003,.003,2)).tolist()
            plan[part].append(s)
    sources={p.name:sha(p) for p in sorted((ROOT/'staged').glob('*.py')) if not p.name.startswith('test_')}
    saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
    saved.update(sources=sources,model_version='visual_perception_v26_calibration_only')
    torch.save(saved,ROOT/'best.pt')
    write('plan.json',plan);write('seed-preparation.json',checks)
    protected={str(p):sha(p) for root in (PARENT,FINISH) for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.pt')}
    write('candidate.json',dict(checkpoint_sha256=sha(ROOT/'best.pt'),parent_checkpoint_sha256=PARENT_SHA,
        plan_sha256=sha(ROOT/'plan.json'),runtime_sources=sources,protected_parent_files=protected,trained=False,
        scope='Camera and label collection; learned weights unchanged. Autonomous queues gated until component validation.'))
    print(json.dumps(dict(counts={k:len(plan[k]) for k in ('pilot','collect','development','test')},checkpoint_sha256=sha(ROOT/'best.pt'))))

if __name__=='__main__':main()
