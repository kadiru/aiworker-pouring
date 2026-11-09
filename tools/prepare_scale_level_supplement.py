"""Preserve failed fits and declare only 36 new intermediate-fill train scenes."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from pour_perception_geometry import calibration_seed_points
from scale_level_supplement import extend
ROOT=Path('/workspace/learning/A-scale-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=15))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not m['trained'] and not read(ROOT/'static-level-fit.json')['passed'] and not (ROOT/'component-selection.json').exists()
    assert read(ROOT/'level-fit-r9/completion.json')['report_sha256']==sha(ROOT/'static-level-fit.json')
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    inventory={}
    for s in plan['collect']:
        if s['study_split']!='train' or not s['name'].startswith('SC_A_static_train_'):continue
        assert sha(Path(s['calibration_particle_seed']))==s['calibration_particle_seed_sha256']
        seed=np.load(s['calibration_particle_seed'])['cup_local_points'];new=dict(s,calibration_initial_height=.35)
        points=calibration_seed_points(new,seed)
        inventory[s['name']]=dict(particles=len(points),initial_positions_sha256=hashlib.sha256(points.tobytes()).hexdigest())
    new=extend(plan,inventory)
    assert not started & set(new['collection_rounds']['r2'])
    archive=ROOT/'collection-r1';archive.mkdir()
    names=('plan.json','candidate.json','collect-geometry.json','holdout-geometry.json','collect-r1-audit.json','collect-r1-manual-review.json',
        'cup-neural-localization.json','scope-selection.json','cup-locator-data-manifest.json','static-level-fit.json','static-level-candidate.pt',
        'level-localization-check.json','fit_scale_level_cnn.py','scale_level_cnn.py','fit_scale_level.py','run_scale_locator_fit.py','run_scale_level_pipeline.py')
    saved={}
    for n in names:shutil.copy2(ROOT/n,archive/n);saved[n]=sha(archive/n);assert saved[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=saved,accepted_locator_checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt')))
    write(ROOT/'plan.json',new)
    m.update(plan_sha256=sha(ROOT/'plan.json'),collection_audit='collect-r2-audit.json',collection_review='collect-r2-manual-review.json',collection_round='r2',stage='intermediate_fill_collection',accepted_locator=False,static_level_accepted=False)
    write(ROOT/'candidate.json',m)
    write(ROOT/'level-supplement-preparation.json',dict(new_cases=36,training=126,validation=18,unused_holdouts=9,plan_sha256=m['plan_sha256'],parent_plan_sha256=saved['plan.json'],inventory=inventory,archive_index_sha256=sha(archive/'index.json'),helper_sha256=sha(Path(__file__)),geometry_required=True,locator_revalidation_required=True,scope='Authored training-only static scenes; seeds cuts are not measured labels. No test or active robot change.'))
    print(json.dumps(dict(new_cases=36,training=126,validation=18,plan_sha256=m['plan_sha256'])))
if __name__=='__main__':main()
