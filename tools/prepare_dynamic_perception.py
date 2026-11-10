"""Create an isolated successor without modifying completed size-study evidence."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
PARENT=Path('/workspace/learning/A-scale-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'candidate.json').exists()
    old=read(PARENT/'candidate.json');selection=read(PARENT/'component-selection.json')
    assert read(PARENT/'static-holdout-evaluation.json')['passed'] and selection['all_size_component_gates_passed']
    assert all(sha(Path(p))==v for p,v in old['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in old['runtime_sources'].items())
    files=['plan.json','best.pt','collect-r2-audit.json','collect-r2-manual-review.json','scope-selection.json',
           'cup-neural-localization.pt','cup-neural-localization.json']
    files+=['fit_scale_level.py','scale_data_admission.py','scale_rim_refiner.py','scale_surface_geometry.py',
            'scale_level_readout.py','scale_level_cnn.py','scale_rim_augmentation.py']
    for n in files:shutil.copy2(PARENT/n,ROOT/n)
    shutil.copy2(PARENT/'selected-static/static-level-candidate.pt',ROOT/'primary-level.pt')
    protected=dict(old['protected_parent_files'])
    for p in PARENT.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.pt','.py','.npz'):protected[str(p)]=sha(p)
    assert not any(p.startswith('/workspace/src/') for p in protected)
    m=dict(old,stage='color_baseline_fit',trained=False,autonomy_accepted=False,protected_parent_files=protected,
           parent_study=str(PARENT),primary_checkpoint_sha256=sha(ROOT/'primary-level.pt'))
    write(ROOT/'candidate.json',m)
    plan=read(ROOT/'plan.json');seed=[]
    sizes=[(h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1)]
    for i,(height,diameter) in enumerate(sizes):
        s=copy.deepcopy(next(s for s in plan['collect'] if s['cup_height_scale']==height and s['cup_diameter_scale']==diameter and s['study_split']=='train' and s['calibration_initial_height']==1.))
        s.update(name=f'SD_A_high_r0_{i:02d}',study_split='high_seed_training',max_seconds=10.,
            calibration_protocol='size_dynamic_v1',calibration_fill=dict(flow_nominal_ml_s=.75,target_height_fraction=.956,start_s=.5,max_feed_s=6.,settle_s=3.))
        seed.append(s)
    protocol=dict(version=1,seed=seed,seed_pilot_names=[seed[i]['name'] for i in (0,4,8)],
        parent_plan_sha256=sha(ROOT/'plan.json'),parent_selection_sha256=sha(PARENT/'component-selection.json'),
        color_fit=dict(feature_sets=['color_geometry','physical_surface'],ridge_penalties=[.001,.01,.1,1.,10.,100.],
            kernel_length_scales=[.5,1.,2.,4.],kernel_ridge_penalties=[.0001,.001,.01,.1]),
        gates=dict(plan['component_gates'],color_validation_mae=.02,color_near_max_error=.035,
            color_near_overestimate=.025,crosscheck_max_disagreement=.035,cpu_gpu_level_difference=.0001,
            high_fill_min=.94,high_fill_max=.98,maximum_lost_particles=5,moving_min_translation_m=.02,moving_min_rotation_deg=15.),
        moving_protocol=dict(duration_s=4.,train='Existing training layouts only; whole episodes',validation='Two existing validation layout families per size with newly declared motion',
            regimes=['Low cup / colored bottle approaching','High cup / empty bottle tilting and withdrawing'],
            reserved='18 fresh low/high moving fixtures, declared before fitting on new images; old nine tests never reused for selection'),
        autonomy_gates=['Reviewed physical seeds','All-size primary and independent color train/validation accuracy',
            'High-fill and moving-view independent tests','Missing/occluded-color rejection','Shared runtime inference parity',
            'All-size full motion/placement clearance','Explicit frozen autonomy acceptance'],
        scope='Known dimensions, original bottle and material A. Authored calibration is never autonomous grasp evidence. No old reserved images enter fitting.')
    write(ROOT/'protocol.json',protocol)
    write(ROOT/'preparation.json',dict(protected_files=len(protected),baseline_training=126,baseline_validation=18,
        baseline_plan_sha256=sha(ROOT/'plan.json'),protocol_sha256=sha(ROOT/'protocol.json'),
        primary_checkpoint_sha256=m['primary_checkpoint_sha256'],helper_sha256=sha(Path(__file__)),autonomy_accepted=False))
    print(json.dumps(dict(created=str(ROOT),high_seed_cases=9,autonomy_accepted=False)))
if __name__=='__main__':main()
