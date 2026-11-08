"""Combine independently validated components; keep parent/controller archives."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
import torch
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'final-selection.json').exists()
    manifest=read(ROOT/'candidate.json');assert not manifest['trained']
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in manifest['protected_parent_files'].items())
    cup=read(ROOT/'cup-component.json');presence=read(ROOT/'presence-component.json')
    assert cup['passed'] and presence['passed']
    assert cup['scope']=='baseline_cup_only' and manifest['validated_cup_scope']=='baseline_only'
    assert sha(ROOT/'cup-component.pt')==cup['checkpoint_sha256'] and sha(ROOT/'presence-component.pt')==presence['checkpoint_sha256']
    assert not any(Path('/workspace/outputs/fountain').glob('*/*PV_A_test_*/parameters.json'))
    assert not any(Path('/workspace/outputs/fountain').glob('*/*PV_A_perception_test_*/parameters.json'))
    round_number=read(ROOT/'plan.json').get('perception_round',0)
    archive=ROOT/('before-component-combination' if round_number==0 else f'before-component-combination-r{round_number}');archive.mkdir()
    for n in ('candidate.json','best.pt'):shutil.copy2(ROOT/n,archive/n)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    c=torch.load(ROOT/'cup-component.pt',map_location='cpu',weights_only=False)
    p=torch.load(ROOT/'presence-component.pt',map_location='cpu',weights_only=False)
    saved.update(c['level']);saved.update(cup_bank=c['localization']['bank'],cup_localization_calibration=c['localization']['calibration'],
        cup_detail_calibration=c['detail_calibration'],wrist_presence_model=p['state'],wrist_presence_roi=p['roi'])
    saved['thresholds']['cup_localization_score']=c['localization']['threshold']
    saved['localization_consensus']['cup']=c['localization']['consensus']
    saved['thresholds']['full_settled_target']=read(ROOT/'plan.json')['component_gates']['stop_target_surface_fraction']
    saved.update(model_version=f'visual_perception_v26_r{round_number}',
        method='Supervised cup-level CNN and optional learned stop-level readout (weighted squared error), wrist bottle-presence CNN (binary cross entropy), and RGB center-heatmap localization (focal heatmap plus smooth-L1 offset loss); retained empty CNN and contextual motion BC/IK; programmed grasp/stop/return. No reinforcement learning.',
        inputs='Task RGB, wrist RGB, cup-detail RGB, measured joints/velocities, fixed camera/robot calibration, own history; no runtime simulator object/particle state.',
        perception_components=dict(cup_sha256=cup['checkpoint_sha256'],presence_sha256=presence['checkpoint_sha256']))
    for name in ('pour_guard_actor.py','pour_perception_study.py','pour_guard_vision.py','pour_cup_locator.py'):manifest['runtime_sources'][name]=sha(ROOT/'staged'/name)
    saved['sources']=manifest['runtime_sources']
    torch.save(saved,ROOT/'best.pt')
    manifest.update(trained=True,component_gates_passed=True,checkpoint_sha256=sha(ROOT/'best.pt'))
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'component-selection.json').write_text(json.dumps(dict(checkpoint_sha256=manifest['checkpoint_sha256'],plan_sha256=sha(ROOT/'plan.json'),
        component_sources={n:sha(ROOT/n) for n in ('cup-component.json','presence-component.json','cup-level-validation.json','cup-localization-validation.json')},
        runtime_sources=manifest['runtime_sources'],validated_cup_scope='baseline_only',scope='Original-cup component-selected candidate only; independent perception holdout and autonomous development still required. Resized cup and bottle autonomy remain gated.'),indent=2)+'\n')
    print(json.dumps(dict(checkpoint_sha256=manifest['checkpoint_sha256'],component_gates_passed=True)))
if __name__=='__main__':main()
