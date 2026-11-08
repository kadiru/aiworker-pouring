"""Declare nine unused original-cup checks before component selection."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from pour_vessel_geometry import cup_geometry,cup_profile
from pour_perception_geometry import calibration_seed_points
from bottle_pour import vessel_mask
from pour_guard_vision import project
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active']
    assert not (ROOT/'component-selection.json').exists()
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');assert manifest['validated_cup_scope']=='baseline_only' and not manifest['trained']
    assert 'deferred_size_holdout' not in plan
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')}
    assert {s['name'] for s in plan['perception_holdout']}.isdisjoint(started)
    archive=ROOT/'before-baseline-holdout';archive.mkdir()
    for n in ('candidate.json','plan.json','geometry-preflight.json'):shutil.copy2(ROOT/n,archive/n)
    plan['deferred_size_holdout']=copy.deepcopy(plan['perception_holdout']);scenes=[];checks=[]
    cal=read(ROOT/'camera-frustum-preflight.json')['calibration']
    for i,(original,cut) in enumerate(zip(plan['perception_holdout'],(.10,.35,.70,.94,1.,.87,.10,1.,.99))):
        s=copy.deepcopy(original);s.update(name=f'PV_A_baseline_perception_{i:02d}',cup_height_scale=1.,cup_diameter_scale=1.,calibration_initial_height=cut)
        s['declared_cup_geometry']=cup_geometry(s)
        with np.load(s['calibration_particle_seed']) as f:points=calibration_seed_points(s,f['cup_local_points'])
        assert vessel_mask(points,cup_profile(s)).all()
        s['cup_initial_ml']=len(points)*s['spacing']**3*1e6
        g=s['declared_cup_geometry'];radius=g['rim_radius_m']+.004;c=np.asarray(s['cup_world'])
        uv=np.array([project(c+[x,y,z],cal) for x in (-radius,radius) for y in (-radius,radius) for z in (0.,g['rim_z_m']+.003)])
        assert (uv>8).all() and (uv<1016).all()
        scenes.append(s);checks.append(dict(name=s['name'],seed_cut=cut,particles=len(points),positions_sha256=hashlib.sha256(points.tobytes()).hexdigest(),frustum_visible=True))
    plan.update(version=plan['version']+1,perception_holdout=scenes)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');manifest['plan_sha256']=sha(ROOT/'plan.json')
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'baseline-holdout-preparation.json').write_text(json.dumps(dict(cases=checks,plan_sha256=manifest['plan_sha256'],helper_sha256=sha(Path(__file__)),
        scope='Nine newly declared original-cup views spanning low through near-full seed cuts. Initial sizes/positions are declared before their images exist. Original nine size holdouts remain unstarted and deferred; autonomous size changes stay gated. Actual settled heights are measured, not assumed from the seed cuts.'),indent=2)+'\n')
    print(json.dumps(dict(baseline_holdouts=9,deferred_size_holdouts=9,plan_sha256=manifest['plan_sha256'])))
if __name__=='__main__':main()
