"""Bounded supervised settled-height regression in the owned 4070 process."""
import copy
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from pour_guard_models import CupEstimator
from pour_guard_vision import locate, plane_point, project
from pour_reliable_data import build_data, calibrate_localization
from pour_settled_model import features, predict
from pour_reliable_study import ROOT, PARENT, PARENT_SHA, PREFIX, plan, RUNTIME_SOURCES


def cover_workspace(saved):
    cal = saved['calibration']; points = {'source': [], 'cup': []}
    for cx, cy, sx, sy in itertools.product((.482, .498), (-.158, -.142), (-.129, -.111), (-.004, .004)):
        points['source'].append(project(np.array([cx - sy, cy + sx, 1.1752]), cal))
        points['cup'].append(project(np.array([cx, cy, 1.078]), cal))
    before = {}
    for kind, uv in points.items():
        bank = saved[kind + '_bank']; before[kind] = list(bank['bounds'])
        a = np.asarray(uv); lo = np.floor(a.min(0) - 6).astype(int); hi = np.ceil(a.max(0) + 6).astype(int)
        bank['bounds'] = [min(before[kind][0], int(lo[0])), min(before[kind][1], int(lo[1])),
            max(before[kind][2], int(hi[0])), max(before[kind][3], int(hi[1]))]
    return dict(before=before, after={k: saved[k + '_bank']['bounds'] for k in points},
        rule='Project all16 corners of the declared fixture domain, add6pixel margin, union with previous search bounds; template kernels and acceptance thresholds unchanged.')


@torch.inference_mode()
def train(state):
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}test*/parameters.json'))
    assert torch.cuda.device_count() == 1 and '4070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4); torch.manual_seed(20261201); plan(); started = time.monotonic()
    assert hashlib.sha256((PARENT / 'best.pt').read_bytes()).hexdigest() == PARENT_SHA
    saved = torch.load(PARENT / 'best.pt', map_location='cpu', weights_only=False)
    state.update(phase='Loading consumed camera episodes for settled-height fitting', learning=dict(stage='loading'))
    coverage = cover_workspace(saved); bundle = build_data(saved); localization = []
    for e in bundle['initial']:
        row = dict(name=e['name'])
        for kind, z, actual in (('source', 1.175, e['source_xyz'][:2]), ('cup', 1.078, e['cup_xyz'][:2])):
            uv, score = locate(e['image'], saved[kind + '_bank'])
            xy = plane_point(uv, z, saved['calibration'])[:2]
            row[kind + '_score'] = score
            row[kind + '_error_mm'] = float(np.linalg.norm(xy - actual) * 1000)
            row[kind + '_accepted'] = score >= saved['thresholds'][kind + '_localization_score']
        localization.append(row)
    loc = dict(coverage=coverage, observations=localization,
        accepted=sum(x['source_accepted'] and x['cup_accepted'] for x in localization),
        max_source_error_mm=max(x['source_error_mm'] for x in localization),
        max_cup_error_mm=max(x['cup_error_mm'] for x in localization))
    (ROOT / 'localization-validation.json').write_text(json.dumps(loc, indent=2) + '\n')
    assert loc['accepted'] == 36 and loc['max_source_error_mm'] < 5. and loc['max_cup_error_mm'] < 5., loc
    consensus, consensus_report = calibrate_localization(saved,bundle['localization_sequences'])
    saved['localization_consensus']=consensus
    model = CupEstimator().cuda().eval(); model.load_state_dict(saved['cup_model'])
    matrices = {}
    for split, d in bundle['data'].items():
        xs = []
        for start in range(0, len(d['target']), 128):
            image = d['image'][start:start + 128].cuda()
            cnn = (model(image) + saved['cup_bias']).clamp(0, 1)
            xs.append(features(image, cnn, d['cup_xy'][start:start + 128].cuda()))
        matrices[split] = torch.cat(xs).double()
    tr, va = bundle['data']['train'], bundle['data']['validation']
    x, vx = matrices['train'], matrices['validation']; y = tr['target'].cuda().double(); vy = va['target'].cuda().double()
    counts = Counter(tr['episode'])
    weights = y.new_tensor([1. / counts[n] for n in tr['episode']]) * torch.where(y > .8, 3., 1.)
    weights /= weights.mean()
    candidates = []
    for columns in ([0, 1, 2, 3], [0, 1, 2, 3, 4], list(range(7)),
            [0, 4] + list(range(7, 21)), list(range(21))):
        a = x[:, columns]; mean = a.mean(0); scale = a.std(0).clamp(min=.001); mean[0] = 0.; scale[0] = 1.
        a = (a - mean) / scale
        for ridge in (.001, .01, .1, 1.):
            penalty = torch.eye(len(columns), device='cuda', dtype=torch.float64) * ridge; penalty[0, 0] = 1e-9
            coef = torch.linalg.solve(a.T @ (a * weights[:, None]) + penalty, a.T @ (y * weights))
            fit = dict(columns=columns, mean=mean.cpu().tolist(), scale=scale.cpu().tolist(), coef=coef.cpu().tolist(), ridge=ridge)
            p = predict(vx, fit); error = p - vy; near = vy > .8
            metrics = dict(mae=float(error.abs().mean()), near_full_mae=float(error[near].abs().mean()),
                near_full_p95_abs=float(torch.quantile(error[near].abs(), .95)),
                near_full_p95_overestimate=float(torch.quantile(error[near], .95)),
                near_full_max_overestimate=float(error[near].max()),
                validation_episodes=len(set(va['episode'])), validation_frames=len(vy), near_full_frames=int(near.sum()))
            score = metrics['near_full_mae'] + .5 * metrics['near_full_p95_abs'] + .1 * metrics['mae']
            candidates.append(dict(fit=fit, metrics=metrics, score=score))
    chosen = min(candidates, key=lambda x: x['score'])
    full_target = .90 + max(.01, chosen['metrics']['near_full_p95_overestimate']) + .005
    assert full_target <= .93, 'Settled prediction uncertainty needs improvement before native use'
    saved['parent_training_step']=saved.get('step');saved['step']=0
    saved.update(model_version='visual_reliable_hierarchy_v16', settled_regressor=chosen['fit'],
        settled_validation=chosen['metrics'], parent_checkpoint_sha256=PARENT_SHA,
        method='Supervised final-settled-height ridge regression on frozen visual features; temporally confirmed RGB localization and flow-limited tilt; retained visual servo, imitation trajectory and empty detector')
    saved['thresholds'].update(full_settled_target=full_target, settled_geometry_tolerance=.10,
        settled_stability_std=.01, trim_increment=.005, trim_min_phase=95.,adaptive_trim=True,
        trim_extra_deg=.25,trim_hard_cap_deg=110.,trim_fast_rate_deg_s=35.,trim_creep_rate_deg_s=35.,trim_total_seconds=1.5,
        trim_seconds=0.,discrete_topup=True,max_pauses=5)
    saved['thresholds'].update(coarse_flow_min_level=.25,coarse_flow_limit_ml_s=18.)
    saved['pour_primitive']['slow_rate_deg_s']=5.
    saved['training_device'] = dict(name=torch.cuda.get_device_name(0), visible_cuda_devices=1)
    saved['settled_training'] = dict(train_episodes=len(set(tr['episode'])), validation_episodes=len(set(va['episode'])),
        train_frames=len(y), validation_frames=len(vy), learned_parameters=len(chosen['fit']['coef']),
        objective='episode-balanced weighted squared error to final settled height + ridge penalty',
        frozen_visual_encoders=True, candidates=len(candidates), data_manifest_sha256=hashlib.sha256((ROOT / 'data-manifest.json').read_bytes()).hexdigest())
    saved['sources'] = {n: hashlib.sha256(Path('/workspace/src', n).read_bytes()).hexdigest() for n in RUNTIME_SOURCES}
    torch.save(saved, ROOT / 'best.partial.pt'); (ROOT / 'best.partial.pt').replace(ROOT / 'best.pt')
    summary = dict(checkpoint_sha256=hashlib.sha256((ROOT / 'best.pt').read_bytes()).hexdigest(),
        parent_checkpoint_sha256=PARENT_SHA, method=saved['method'], training=saved['settled_training'],
        device=saved['training_device'], chosen=dict(columns=chosen['fit']['columns'], ridge=chosen['fit']['ridge'], metrics=chosen['metrics']),
        thresholds=saved['thresholds'], localization=loc, localization_consensus=consensus_report,
        motion_rate_deg_s=dict(fast=saved['pour_primitive']['fast_rate_deg_s'],slow=saved['pour_primitive']['slow_rate_deg_s']), elapsed_s=time.monotonic() - started,
        candidates=[dict(columns=c['fit']['columns'], ridge=c['fit']['ridge'], metrics=c['metrics'], score=c['score']) for c in candidates],
        unchanged_components=['cup CNN weights', 'source detector weights', 'template kernels', 'motor primitive path weights (tilt rate reduced)', 'empty confirmation gates'],
        no_privileged_runtime_inputs=True, data_manifest_sha256=saved['settled_training']['data_manifest_sha256'], sources=saved['sources'])
    (ROOT / 'model-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    state.update(phase='Settled-height fitting complete', learning=dict(stage='fit_complete', elapsed_s=summary['elapsed_s'], metrics=chosen['metrics']))
    del model, bundle, matrices, x, vx, y, vy; torch.cuda.empty_cache()
    return str(ROOT / 'best.pt')
