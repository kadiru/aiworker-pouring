"""Idle-only selection of measured impact settings and programmed side return.

The learned tensor/components are retained exactly. This is control/physics
calibration, not a new behavioral-cloning or reinforcement-learning fit.
"""
import argparse,copy,hashlib,io,json,shutil,sys,urllib.request
from pathlib import Path
import torch
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,r):
    temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(r,indent=2)+'\n');temporary.replace(p)
def digest(value):
    stream=io.BytesIO();torch.save(value,stream);return hashlib.sha256(stream.getvalue()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--variant',required=True);a.add_argument('--full-target',type=float,required=True);a.add_argument('--round',type=int,default=0);a=a.parse_args()
assert .92<=a.full_target<=.95 and not (ROOT/'final-selection.json').exists()
h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active'] and not h.get('error')
m=read(ROOT/'candidate.json');assert sha(ROOT/'best.pt')==m['checkpoint_sha256'] and sha(ROOT/'plan.json')==m['plan_sha256']
assert all(sha(Path('/workspace/src')/n)==d for n,d in m['runtime_sources'].items())
plan=read(ROOT/'plan.json');report=read(ROOT/'impact-outcomes.json');assert len(report['cases'])==12
selection=next(r for r in report['cases'] if r['name']==a.variant);assert selection['zero_measured_loss']
assert report['baseline_reproduces_loss'],'A non-reproducing microbenchmark cannot authorize an asserted repair'
chosen=next(s for s in plan['impact'] if s['name']==a.variant)
started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*SR_A_*/parameters.json')}
assert not started&{s['name'] for s in plan['development']+plan['test']}
archive=ROOT/f'rounds/before-candidate-r{a.round}';archive.mkdir(parents=True,exist_ok=False)
for n in ('best.pt','candidate.json','plan.json','impact-outcomes.json','recording-audit.json'):shutil.copy2(ROOT/n,archive/n)
saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
unchanged={k:digest(v) for k,v in saved.items() if k not in ('thresholds','side_return','side_return_version')}
old_thresholds=copy.deepcopy(saved['thresholds']);old_target=saved['thresholds']['full_settled_target'];saved['thresholds']=dict(saved['thresholds'],full_settled_target=a.full_target)
saved['side_return']=dict(upright_seconds=2.4,place_seconds=1.8,retreat_seconds=2.6,preferred_side='robot_right')
saved['side_return_version']=1
assert all(digest(saved[k])==v for k,v in unchanged.items())
temporary=ROOT/f'candidate-r{a.round}.tmp.pt';assert not temporary.exists();torch.save(saved,temporary);loaded=torch.load(temporary,map_location='cpu',weights_only=False)
assert all(digest(loaded[k])==v for k,v in unchanged.items())
assert {k:v for k,v in loaded['thresholds'].items() if k!='full_settled_target'}=={k:v for k,v in old_thresholds.items() if k!='full_settled_target'}
assert not torch.cuda.is_initialized();temporary.replace(ROOT/'best.pt')
physics_keys=('viscosity','cohesion','surface_tension','friction','adhesion','damping','vorticity','iterations','max_depenetration_velocity','cfl_coefficient')
physics={k:chosen[k] for k in physics_keys if k in chosen}
for s in plan['development']+plan['test']:s.update(physics)
plan.update(version=plan['version']+1,selected_impact_variant=a.variant,selected_physics=physics,
    selection_scope='Cold-start impact comparison plus forthcoming full autonomous development; no held-out task outcome has been used.')
write(ROOT/'plan.json',plan)
m.update(checkpoint_sha256=sha(ROOT/'best.pt'),plan_sha256=sha(ROOT/'plan.json'),stage='development_candidate',candidate_round=a.round,
    method='Retained supervised RGB perception and contextual motion BC; calibrated stop target and programmed simultaneous side-return IK. No new fit.',
    selected_physics=physics)
write(ROOT/'candidate.json',m)
record=dict(candidate=m,old_target=old_target,new_target=a.full_target,side_return=saved['side_return'],
    unchanged_component_digests=unchanged,impact_outcomes_sha256=sha(ROOT/'impact-outcomes.json'),chosen_impact_result=selection,
    helper_sha256=sha(Path(__file__)),scope=m['method'])
write(ROOT/f'candidate-selection-r{a.round}.json',record);print(json.dumps(dict(checkpoint_sha256=m['checkpoint_sha256'],physics=physics,full_target=a.full_target)))
