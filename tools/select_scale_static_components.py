"""Freeze separately fitted static components before once-only holdout views.

The deployed robot checkpoint is preserved. This accepts only the declared
static component gates, not higher-fill, occluded or autonomous task behavior.
"""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from scale_data_admission import collection_paths
ROOT=Path('/workspace/learning/A-scale-20261109-v1')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output=ROOT/'component-selection.json';assert not output.exists()
    m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json')
    audit_path,review_path=collection_paths(ROOT,m)
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(Path(p))==d for p,d in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in m['runtime_sources'].items())
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error')
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started.intersection(s['name'] for s in plan['holdout'])
    loc=read(ROOT/'cup-neural-localization.json');level=read(ROOT/'static-level-fit.json')
    assert loc['passed'] and loc['scope']=='all_nine_declared_cup_sizes' and level['passed']
    assert loc['plan_sha256']==level['plan_sha256']==m['plan_sha256']
    assert loc['audit_sha256']==level['audit_sha256']==sha(audit_path)
    assert loc['manual_review_sha256']==level['manual_review_sha256']==sha(review_path)
    assert level['localization_checkpoint_sha256']==loc['checkpoint_sha256']==sha(ROOT/'cup-neural-localization.pt')
    assert level['parent_checkpoint_sha256']==m['checkpoint_sha256']
    assert level['checkpoint_sha256']==sha(ROOT/'static-level-candidate.pt')
    assert loc['module_sha256']==level['localization_module_sha256']==sha(ROOT/'scale_rim_refiner.py')
    assert level['level_module_sha256']==sha(ROOT/'scale_level_readout.py')
    assert level['surface_module_sha256']==sha(ROOT/'scale_surface_geometry.py')
    if 'cnn_module_sha256' in level:assert level['cnn_module_sha256']==sha(ROOT/'scale_level_cnn.py')
    selected=ROOT/'selected-static';selected.mkdir()
    files={}
    for name in ('cup-neural-localization.pt','static-level-candidate.pt','cup-neural-localization.json',
                 'static-level-fit.json','level-localization-check.json','scope-selection.json'):
        shutil.copy2(ROOT/name,selected/name);files[name]=dict(path=str(selected/name),sha256=sha(selected/name))
    report=dict(checkpoint_sha256=m['checkpoint_sha256'],plan_sha256=m['plan_sha256'],runtime_sources=m['runtime_sources'],
        all_size_component_gates_passed=True,components=files,holdout_names=[s['name'] for s in plan['holdout']],
        evaluation_helper_sha256=sha(ROOT/'evaluate_scale_static_holdout.py'),evaluation_observation_frames=12,
        localization_module_sha256=sha(ROOT/'scale_rim_refiner.py'),
        level_module_sha256=sha(ROOT/'scale_level_readout.py'),
        surface_module_sha256=sha(ROOT/'scale_surface_geometry.py'),
        cnn_module_sha256=level.get('cnn_module_sha256'),
        primary_model='selected_static_candidate',comparison_model='frozen_parent_no_selection',
        model_activated=False,autonomous_use_accepted=False,helper_sha256=sha(Path(__file__)),
        scope='Static known-size cup localization and level estimation in the measured collection range only. Freeze before nine unused near-full static holdouts; parent robot checkpoint remains unchanged. Independent color, >=94% fill, held-bottle, dynamic and task gates are not accepted.')
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    m.update(stage='static_components_selected',trained=True,active_checkpoint_retained=True,
        component_scope='static_known_sizes_only',component_selection_sha256=sha(output),
        method='Separately trained static cup locator and level readout; active parent robot checkpoint retained. Resized autonomy remains gated.')
    temp=ROOT/'candidate.tmp';temp.write_text(json.dumps(m,indent=2)+'\n');temp.replace(ROOT/'candidate.json')
    print(json.dumps(dict(selected=True,holdouts=len(plan['holdout']),model_activated=False)))


if __name__=='__main__':main()
