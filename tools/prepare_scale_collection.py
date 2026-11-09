"""Declare balanced static perception data after all nine physical seeds pass."""
import copy
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from pour_perception_geometry import calibration_seed_points
from pour_guard_vision import project
ROOT=Path('/workspace/learning/A-scale-20261109-v1')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error')
    plan=read(ROOT/'plan.json');m=read(ROOT/'candidate.json')
    assert plan['version']==3 and not plan['collect'] and not plan['holdout'] and not m['trained']
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    seeds={}
    for revision in ('r1','r2'):
        audit=read(ROOT/f'pilot-{revision}-audit.json');review=read(ROOT/f'pilot-{revision}-manual-review.json')
        assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ROOT/f'pilot-{revision}-audit.json')
        for r in audit['records']:
            assert r['label']['accepted'] and r['high_fill_coverage']
            spec=read(Path(r['path']).parent/'parameters.json');key=(spec['cup_height_scale'],spec['cup_diameter_scale'])
            assert key not in seeds and sha(Path(r['settled_seed_path']))==r['settled_seed_sha256']
            seeds[key]=(spec,r)
    assert len(seeds)==9
    reference_path=Path('/workspace/learning/A-wide-20261105-v1/frozen-plan.json')
    refs=read(reference_path)['demonstrations']
    train=[s for s in refs if s['study_split']=='train'];validation=[s for s in refs if s['study_split']=='validation']
    collection=[];holdout=[];checks=[];rng=np.random.default_rng(26110947)
    for size_index,(key,(base,record)) in enumerate(sorted(seeds.items())):
        with np.load(record['settled_seed_path'],allow_pickle=False) as f:points=f['cup_local_points'].copy()
        # Rotate fills across layouts so a fixed table location or swapped
        # arrangement does not always imply the same level across sizes.
        cuts=np.roll(np.asarray((.22,.5,.78,1.)),size_index%4).tolist()
        variants=[('train',j,train[index],cut) for j,(index,cut) in enumerate(zip((0,3,4,7),cuts))]
        val_cuts=(.3,1.) if size_index%2==0 else (1.,.3)
        variants += [('validation',j,validation[index],cut) for j,(index,cut) in enumerate(zip((0,3),val_cuts))]
        variants += [('holdout',0,validation[4 if size_index%2==0 else 5],1.)]
        for split,j,layout,cut in variants:
            s=copy.deepcopy(base);s.pop('calibration_fill',None)
            s.update(name=f'SC_A_static_{split}_{size_index:02d}_{j:02d}',study_split=split,
                calibration_version=4,calibration_particle_seed=record['settled_seed_path'],
                calibration_particle_seed_sha256=record['settled_seed_sha256'],calibration_initial_height=cut,
                max_seconds=2.5,cup_world=copy.deepcopy(layout['cup_world']),source_xy=copy.deepcopy(layout['source_xy']),
                layout_reference=layout['name'],swapped=layout.get('swapped',False),
                layout_label=layout.get('layout_label',layout['name']),initialization_revision='size_specific_native_settled_seed_cut')
            if split=='holdout':
                offset=rng.uniform(-.002,.002,2);s['cup_world'][:2]=(np.asarray(s['cup_world'][:2])+offset).tolist()
                s['holdout_fixture_jitter_m']=offset.tolist()
            selected=calibration_seed_points(s,points);s['cup_initial_ml']=len(selected)*s['spacing']**3*1e6
            cal=copy.deepcopy(record['cameras']['cup_detail']['calibration']);cal['K']=(np.asarray(cal['K'])*np.array([[384/1024],[384/1024],[1.]])).tolist()
            uv=project(np.asarray(s['cup_world'])+[0.,0.,s['declared_cup_geometry']['rim_z_m']],cal)
            assert (uv>=36).all() and (uv<=348).all(),('Locator augmentation frustum',s['name'],uv)
            checks.append(dict(name=s['name'],size=list(key),split=split,seed_cut=cut,particles=len(selected),
                               source_seed_particles=len(points),rim_uv_384=uv.tolist(),initial_particle_positions_sha256=hashlib.sha256(selected.tobytes()).hexdigest()))
            (holdout if split=='holdout' else collection).append(s)
    assert len(collection)==54 and len(holdout)==9
    archive=ROOT/'seed-r2';archive.mkdir()
    for name in ('plan.json','candidate.json','pilot-geometry.json','pilot-r2-audit.json','pilot-r2-manual-review.json'):
        shutil.copy2(ROOT/name,archive/name)
    plan.update(version=4,collect=collection,holdout=holdout,primary_perception_scope='all_nine_declared_cup_sizes',
        scope='54 authored static perception episodes: four train and two validation per size, with separate full episodes and known varied/swapped layout families. Nine unused static holdouts declared before fitting. Parked arm and original bottle; not task demonstrations or unknown-size estimation.')
    plan['component_gates']['localization_negative_margin']=.015
    write(ROOT/'plan.json',plan);m.update(plan_sha256=sha(ROOT/'plan.json'),stage='static_collection',seed_and_camera_accepted=True)
    write(ROOT/'candidate.json',m)
    write(ROOT/'collection-preparation.json',dict(cases=checks,reference_plan_sha256=sha(reference_path),helper_sha256=sha(Path(__file__)),
        training_episodes=36,validation_episodes=18,unused_holdout_episodes=9,
        next_gate='Native geometry and camera/stability audits before fitting. Each size/split must retain actual low and >=90% labels. Higher-fill and held-bottle coverage remain separate concerns.'))
    print(json.dumps(dict(train=36,validation=18,unused_holdout=9,plan_sha256=m['plan_sha256'],trained=False)))


if __name__=='__main__':main()
