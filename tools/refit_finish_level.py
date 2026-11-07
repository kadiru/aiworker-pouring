"""Prepare a separate 4070 level-refit candidate from audited development.

Never replaces the active model/sources, queues a run, or touches reserved
images. Reuses the historical data builder; new development views are training
data, explicitly not validation or unbiased performance evidence.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, '/workspace/src')
import pour_finish_train as original_fit
from pour_finish_study import ROOT, OUTPUTS, sha
from pour_guard_models import CupEstimator
from pour_guard_vision import cup_crop
from pour_settled_model import features, predict
from validate_pour_recording import h5py, decode
from calibrate_finish_target import equal


def read(p):
    return json.loads(p.read_text())


def write(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + '\n')


def settled_supervision(output):
    expected = {r['name']:r for r in read(ROOT / 'calibration-plan.json')['cases']}
    groups, records = {}, []
    for path in sorted(OUTPUTS.glob('*/*FCAL*/samples.json')):
        spec = read(path.parent / 'parameters.json')
        if spec['name'] not in expected:
            continue
        assert spec == expected[spec['name']]
        rows = read(path)
        last = rows[-1]
        assert not last['ever_table_ml'] and not last['airborne_ml']
        record = dict(name=spec['name'], split=spec['study_split'], cup_ml=last['cup_ml'],
            height=last['cup_fill_fraction'], samples_sha256=sha(path))
        records.append(record)
        if spec['study_split'] == 'train':
            groups.setdefault(last['cup_ml'], []).append(last['cup_fill_fraction'])
    assert len(records) == 51 and {r['name'] for r in records} == set(expected)
    x = np.array(sorted(groups))
    y = np.array([np.mean(groups[v]) for v in x])
    assert np.all(np.diff(y) > 0)
    def label(volume):
        assert volume >= x[0]
        if volume > x[-1]:
            return min(1., float(y[-1]+(volume-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])))
        return float(np.interp(volume, x, y))
    validation = [dict(r, predicted_height=label(r['cup_ml']), error=label(r['cup_ml'])-r['height']) for r in records if r['split']=='validation']
    assert len(validation) == 12 and max(abs(r['error']) for r in validation) < .005
    report = dict(x=x.tolist(), y=y.tolist(), records=records, validation=validation,
        scope='Training-only static inventory-to-settled-height interpolation, with linear upper extrapolation capped at 1. Used solely for supervised labels of pre-spill training pauses; no inventory input at runtime. Existing validation image labels remain unchanged, including their transient-height limitations.')
    write(output, 'settled-target-supervision.json', report)
    return label, float(x[0])


def relabel_historical_training_pauses(data, manifest, label, minimum):
    count = 0
    for record in manifest['consumed_historical_records']:
        if record['split'] != 'train':
            continue
        positions = [i for i,n in enumerate(data['train']['episode']) if n == record['name']]
        assert len(positions) == len(record['frames'])
        rows = read(Path(record['path']).parent / 'samples.json')
        for i, frame in zip(positions, record['frames']):
            observed = rows[frame['index']]
            if frame['view'] != 'settle' or observed['cup_ml'] < minimum or observed['ever_table_ml']:
                continue
            frame.update(original_target=frame['target'], target=label(observed['cup_ml']),
                target_source='static_train_inventory_to_settled_height', nominal_cup_ml=observed['cup_ml'])
            data['train']['target'][i] = frame['target']
            count += 1
    return count


def add_development(data, saved, integrity, label, minimum):
    records = []
    arrays = {k: [] for k in ('image', 'target', 'cup_xy', 'episode')}
    quantum = .0018**3*1e6
    indexed = {r['name']: r for r in integrity['files']}
    for name, identity in sorted(indexed.items()):
        if not name.startswith('FN30_A_dev_'):
            continue
        path = OUTPUTS / identity['relative_path']
        assert path.name == 'episode.hdf5' and sha(path) == identity['sha256']
        spec = read(path.parent / 'parameters.json')
        assert spec['development_round'] in (0, 1, 2)
        rows = read(path.parent / 'samples.json')
        decisions = read(path.parent / 'policy-decisions.json')
        xy = next(r['cup_xy'] for r in decisions if r.get('cup_xy') is not None)
        assert len(rows) == len(decisions)
        selected = {}
        i = 0
        while i < len(rows):
            if decisions[i]['mode'] != 'Pour control: settle':
                i += 1
                continue
            start = i
            while i < len(rows) and decisions[i]['mode'] == 'Pour control: settle':
                i += 1
            end = i-1
            if rows[end]['t']-rows[start]['t'] < .65 or rows[end]['ever_table_ml']:
                continue
            plateau = rows[-1] if (not rows[-1]['ever_table_ml'] and
                abs(rows[end]['cup_ml']-rows[-1]['cup_ml']) <= 8*quantum) else rows[end]
            for j in range(start, end+1, 2):
                if rows[j]['t']-rows[start]['t'] >= .5 and abs(rows[j]['cup_ml']-plateau['cup_ml']) <= 8*quantum:
                    target = label(plateau['cup_ml']) if plateau is not rows[-1] and plateau['cup_ml'] >= minimum else plateau['cup_fill_fraction']
                    selected[j] = (target, 'settle')
        if not rows[-1]['ever_table_ml']:
            for j in range(0, len(rows), 3):
                if decisions[j]['mode'] in ('Setting bottle down', 'Releasing bottle', 'Holding completed pose') and abs(rows[j]['cup_ml']-rows[-1]['cup_ml']) <= 8*quantum:
                    selected[j] = (rows[-1]['cup_fill_fraction'], 'final')
        with h5py.File(path) as f:
            assert f.attrs['status'] == 'complete'
            d = f['data/demo_0']
            for j, (target, view) in selected.items():
                image = cup_crop(decode(d['next_obs/task_image'][j], 'jpeg95'), xy, saved['calibration'])
                for k, value in dict(image=image, target=target, cup_xy=xy, episode=name).items():
                    arrays[k].append(value)
        records.append(dict(name=name, path=str(path), sha256=identity['sha256'], split='train',
            former_split='consumed_development', development_round=spec['development_round'],
            frames=[dict(index=j, target=v[0], view=v[1]) for j, v in selected.items()],
            pause_target_scope='Measured final height for a final-inventory plateau; otherwise static-training inventory-to-settled-height mapping where covered, or measured low-level plateau height.',
            excluded_after_measured_loss=True,
            exclusion_reason=None if selected else 'No eligible stable view before measured loss'))
    assert len(records) == 24 and all(sum(r['development_round'] == n for r in records) == 8 for n in (0, 1, 2))
    assert arrays['target']
    for k, values in arrays.items():
        if k == 'episode':
            data['train'][k] += values
        else:
            data['train'][k] = torch.cat((data['train'][k], torch.from_numpy(np.asarray(values, dtype=np.uint8 if k == 'image' else np.float32))))
    assert not set(data['train']['episode']) & set(data['validation']['episode'])
    assert all(not name.startswith('FN30_A_') for name in data['validation']['episode'])
    assert all(not name.startswith('FN30_A_test_') for part in data.values() for name in part['episode'])
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--withdrawal', type=Path, required=True)
    p.add_argument('--helper-source-commit', required=True)
    a = p.parse_args()
    parent = 'bc9dc0e2d0afd2d090ff36f283be2276a9090f5fc48d3bf8290291dfcf2ee81e'
    assert len(a.helper_source_commit) == 40 and all(c in '0123456789abcdef' for c in a.helper_source_commit)
    assert a.output.parent == ROOT and not a.output.exists()
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(OUTPUTS.glob('*/*FN30_A_test*/parameters.json'))
    with urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=10) as response:
        h = json.load(response)
    assert not h['queue'] and not (h.get('recording') or {}).get('active')
    assert not h.get('error') and not h.get('learning_error')
    model = read(ROOT / 'model-summary.json')
    assert sha(ROOT / 'best.pt') == model['checkpoint_sha256'] == parent
    for n, digest in model['runtime_sources'].items():
        assert sha(Path('/workspace/src') / n) == digest, n
    manifest = read(a.manifest)
    geometry = read(ROOT / 'late-drip-clearance-r3.json')
    assert manifest['candidate_for_parent_checkpoint_sha256'] == parent
    assert geometry['candidate_runtime_sources'] == manifest['files']
    assert geometry['complete'] and all(r['ready'] for r in geometry['cases'])
    review = read(a.review)
    integrity = read(ROOT / 'development-recording-integrity.json')
    assert review['checkpoint_sha256'] == parent and review['manual_review'] == 'inspected'
    assert len(review['cases']) == 8 and all(r['manual_review'] == 'inspected' and r['name'].endswith('_r2') for r in review['cases'])
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    indexed = {r['name']: r for r in integrity['files']}
    assert all(r['source_sha256'] == indexed[r['name']]['sha256'] for r in review['cases'])
    motion = read(a.withdrawal)
    assert {r['name'] for r in motion['cases']} == {r['name'] for r in review['cases']}
    for r in motion['cases']:
        assert r['source_sha256'] == indexed[r['name']]['sha256']
        assert r['longest_sampled_upright_near_stationary_over_cup_s'] <= .35 and r['post_release_gripper_contact_frames'] == 0
    assert torch.cuda.device_count() == 1 and '4070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    a.output.mkdir()
    for n in ('frozen-plan.json', 'calibration-plan.json'):
        shutil.copy2(ROOT / n, a.output / n)
    original_fit.ROOT = a.output
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    original = copy.deepcopy(saved)
    started = time.monotonic()
    data = original_fit.build(saved)
    data_manifest = read(a.output / 'data-manifest.json')
    label, minimum = settled_supervision(a.output)
    relabeled = relabel_historical_training_pauses(data, data_manifest, label, minimum)
    additions = add_development(data, saved, integrity, label, minimum)
    data_manifest['consumed_historical_records'] += additions
    data_manifest.update(consumed_development_rounds=[0, 1, 2],
        settled_supervision_sha256=sha(a.output / 'settled-target-supervision.json'), historical_training_pause_labels_replaced=relabeled,
        scope='Historical train/validation split retained. Twenty-four closed FN30 development records explicitly consumed for training; only eligible zero-loss views included. No FN30 development in validation and no reserved FN30 test images used.')
    write(a.output, 'data-manifest.json', data_manifest)
    fit, diagnostic = fit_level(data, saved, a.output)
    saved['settled_regressor'] = fit
    saved['settled_validation'] = diagnostic['chosen']['metrics']
    saved['thresholds']['full_settled_target'] = diagnostic['target']
    saved['late_mouth_shift_m'] = .008
    saved['sources'] = manifest['files']
    saved['parent_checkpoint_sha256'] = parent
    saved['model_version'] = 'visual_finish_hierarchy_v24'
    saved['method'] = 'Supervised RGB settled-level ridge and residual MLP refit with consumed development views; retained localization, empty CNN and contextual motion behavioral cloning with IK; programmed stop and direct return with late-pour lip alignment.'
    allowed = {'settled_regressor','settled_validation','thresholds','late_mouth_shift_m','sources','parent_checkpoint_sha256','model_version','method'}
    assert equal({k:v for k,v in original.items() if k not in allowed}, {k:v for k,v in saved.items() if k not in allowed})
    assert equal({k:v for k,v in original['thresholds'].items() if k != 'full_settled_target'}, {k:v for k,v in saved['thresholds'].items() if k != 'full_settled_target'})
    torch.save(saved, a.output / 'best.pt')
    assert equal(saved, torch.load(a.output / 'best.pt', map_location='cpu', weights_only=False))
    checkpoint = sha(a.output / 'best.pt')
    refit = dict(parent_checkpoint_sha256=parent, checkpoint_sha256=checkpoint,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
        device=torch.cuda.get_device_name(0), source_helper_sha256=sha(Path(__file__)),
        source_helper_commit=a.helper_source_commit,
        historical_builder_sha256=sha(Path(original_fit.__file__)),
        source_manifest_sha256=sha(a.manifest), runtime_source_commit=manifest['source_commit'],
        integrity_sha256=sha(ROOT / 'development-recording-integrity.json'),
        settled_supervision_sha256=sha(a.output / 'settled-target-supervision.json'),
        review_sha256=sha(a.review), withdrawal_sha256=sha(a.withdrawal),
        consumed_development_records=[dict(name=r['name'], sha256=r['sha256'], frames=len(r['frames'])) for r in additions],
        old_full_target=model['thresholds']['full_settled_target'], new_full_target=diagnostic['target'],
        old_fitted_sources=model['sources'], unchanged_components=['localizer','cup_CNN','empty_CNN','motion_BC','physics','grasp logic','return timing'],
        scope='Candidate only. Refit changes the settled regressor and validation-derived full threshold; adds 8 mm late lip alignment. No active deployment, native success or reserved result claimed.')
    updated = copy.deepcopy(model)
    updated.update(**diagnostic, checkpoint_sha256=checkpoint, parent_candidate_checkpoint_sha256=parent,
        sources=manifest['files'], runtime_sources=manifest['files'], thresholds=saved['thresholds'],
        data_manifest_sha256=sha(a.output / 'data-manifest.json'), method=saved['method'],
        late_mouth_shift_m=.008, level_refit=refit,
        late_drip_alignment=dict(shift_m=.008, shift_angle_range_deg=[116.,122.], geometry_sha256=sha(ROOT / 'late-drip-clearance-r3.json'),
            scope='Programmed geometry adjustment combined with refit; dynamic validation pending.'),
        device=torch.cuda.get_device_name(0), elapsed_s=refit['elapsed_s'])
    write(a.output, 'level-refit.json', refit)
    write(a.output, 'model-summary.json', updated)
    assert sha(ROOT / 'best.pt') == parent, 'Active checkpoint must remain unchanged'
    print(json.dumps(dict(candidate_checkpoint_sha256=checkpoint, diagnostic=diagnostic, active_unchanged=True)), flush=True)


@torch.inference_mode()
def fit_level(data, saved, output):
    model=CupEstimator().cuda().eval();model.load_state_dict(saved['cup_model']);matrices={}
    for split,d in data.items():
        xs=[]
        for i in range(0,len(d['target']),128):
            a=d['image'][i:i+128].cuda();cnn=(model(a)+saved['cup_bias']).clamp(0,1)
            xs.append(features(a,cnn,d['cup_xy'][i:i+128].cuda()).double())
        matrices[split]=torch.cat(xs)
    x,vx=matrices['train'],matrices['validation'];y=data['train']['target'].cuda().double();vy=data['validation']['target'].cuda().double()
    count=Counter(data['train']['episode']);w=y.new_tensor([1/count[n] for n in data['train']['episode']])*torch.where(y>.8,3.,1.)
    recent=torch.tensor([n.startswith(('XW30_A_','FCAL_A_','FN30_A_dev_')) for n in data['train']['episode']],device='cuda')
    wideval=torch.tensor([n.startswith(('XW30_A_','FCAL_A_','FN30_A_dev_')) for n in data['validation']['episode']],device='cuda');near=vy>.8
    candidates=[]
    feature_sets=[list(range(21)),list(range(60)),list(range(340)),list(range(21))+list(range(340,604)),list(range(604)),
                  list(range(21))+list(range(604,x.shape[1])),list(range(x.shape[1]))]
    for weight in (2.,8.,32.):
        weights=w*torch.where(recent,weight,1.);weights/=weights.mean()
        for columns in feature_sets:
            a=x[:,columns];mean=a.mean(0);scale=a.std(0).clamp(min=.001);mean[0]=0;scale[0]=1;a=(a-mean)/scale
            for ridge in (.1,1.,10.,100.):
                reg=torch.eye(len(columns),device='cuda',dtype=torch.float64)*ridge;reg[0,0]=1e-9
                coef=torch.linalg.solve(a.T@(a*weights[:,None])+reg,a.T@(y*weights))
                fit=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),coef=coef.cpu().tolist(),ridge=ridge)
                for bias in (0.,-.005,-.01,-.015,-.02):
                    fit=dict(fit,output_bias=bias)
                    error=predict(vx,fit)-vy;mask=wideval&near
                    metrics=dict(mae=float(error.abs().mean()),near_mae=float(error[near].abs().mean()),wide_near_mae=float(error[mask].abs().mean()),
                        wide_near_max=float(error[mask].abs().max()),wide_near_overestimate=float(error[mask].max()),near_p95=float(torch.quantile(error[near].abs(),.95)))
                    candidates.append(dict(fit=fit,weight=weight,metrics=metrics,score=metrics['wide_near_mae']+metrics['near_mae']+.3*metrics['wide_near_max']))
    # Ridge leaves opposite perspective-dependent errors. Fit a bounded
    # nonlinear residual on training episodes; validation selects the epoch.
    base=min(candidates,key=lambda c:c['score'])['fit']
    for columns in (list(range(7))+list(range(340,367)),list(range(7))+list(range(340,367))+list(range(604,x.shape[1]))):
        with torch.inference_mode(False),torch.enable_grad():
            xx=x[:,columns].float().clone();vv=vx[:,columns].float().clone()
            mean=xx.mean(0);scale=xx.std(0).clamp(min=.001);xx=(xx-mean)/scale;vv=(vv-mean)/scale
            baseline=predict(x,base).float().clone();vbase=predict(vx,base).float().clone()
            yy=y.float().clone();vyy=vy.float().clone();weights=(w*torch.where(recent,16.,1.)).float().clone();weights/=weights.mean()
            for width in (32,64):
                torch.manual_seed(20261106+width)
                net=torch.nn.Sequential(torch.nn.Linear(len(columns),width),torch.nn.Tanh(),torch.nn.Linear(width,16),torch.nn.Tanh(),torch.nn.Linear(16,1)).cuda()
                torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
                opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
                for step in range(1,1201):
                    opt.zero_grad();delta=net(xx)[:,0];loss=((baseline+delta-yy).square()*weights).mean()+.02*delta.square().mean();loss.backward();opt.step()
                    if step%20:continue
                    with torch.no_grad():
                        error=(vbase+net(vv)[:,0]).clamp(0,1)-vyy;mask=wideval&near
                        metrics=dict(mae=float(error.abs().mean()),near_mae=float(error[near].abs().mean()),wide_near_mae=float(error[mask].abs().mean()),
                            wide_near_max=float(error[mask].abs().max()),wide_near_overestimate=float(error[mask].max()),near_p95=float(torch.quantile(error[near].abs(),.95)))
                        layers=[dict(weight=l.weight.cpu().tolist(),bias=l.bias.cpu().tolist()) for l in net if isinstance(l,torch.nn.Linear)]
                        fitted=dict(base,residual_mlp=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),layers=layers))
                        candidates.append(dict(fit=fitted,weight=16.,residual_inputs=len(columns),residual_width=width,optimizer_steps=step,metrics=metrics,
                            score=metrics['wide_near_mae']+metrics['near_mae']+.3*metrics['wide_near_max']))
    consumed = torch.tensor([n.startswith('FN30_A_dev_') for n in data['train']['episode']], device='cuda') & (y>.8)
    assert consumed.any()
    for candidate in candidates:
        error=predict(x, candidate['fit'])[consumed]-y[consumed]
        candidate['metrics'].update(consumed_development_near_mae=float(error.abs().mean()),
            consumed_development_near_max=float(error.abs().max()),
            consumed_development_near_underestimate=float(error.min()))
    write(output, 'level-candidates.json', candidates)
    eligible=[c for c in candidates if c['metrics']['wide_near_max']<.04
        and c['metrics']['consumed_development_near_max']<.03
        and c['metrics']['consumed_development_near_underestimate']>-.025]
    chosen=min(eligible or candidates, key=lambda c:c['score'])
    metrics=chosen['metrics']
    target=.90+max(.01,metrics['wide_near_overestimate'])+.002
    report=dict(status='component_failed', chosen={k:v for k,v in chosen.items() if k!='fit'}, target=target,
        training={split:dict(episodes=len(set(d['episode'])), frames=len(d['target'])) for split,d in data.items()},
        parameters=len(chosen['fit']['coef'])+sum(len(l['bias'])+sum(len(row) for row in l['weight']) for l in chosen['fit'].get('residual_mlp',{}).get('layers',[])),
        selection_scope='Historical validation selects among fits meeting the declared consumed-development residual screen; development residuals are training error, not generalization evidence.')
    write(output, 'fit-diagnostic.json', report)
    assert eligible and target<=.94, 'Level refit component needs inspection before native development'
    report['status']='fit_complete'
    write(output, 'fit-diagnostic.json', report)
    return chosen['fit'], report



if __name__ == '__main__':
    main()
