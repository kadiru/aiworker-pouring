"""Expanded-position supervised calibration and motion imitation on the 4070."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import torch
from pour_guard_models import CupEstimator
from pour_reliable_data import calibrate_localization
from pour_workspace_data import sequences, fit_banks, build_data, probe_empty_detector
from pour_workspace_motion import fit as fit_motion
from pour_settled_model import features, predict
from pour_workspace_study import ROOT, PARENT, PARENT_SHA, PREFIX, plan, RUNTIME_SOURCES


@torch.inference_mode()
def train(state):
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}test*/parameters.json'))
    assert torch.cuda.device_count() == 1 and '4070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4); torch.manual_seed(20261201); plan(); started = time.monotonic()
    assert hashlib.sha256((PARENT / 'best.pt').read_bytes()).hexdigest() == PARENT_SHA
    saved = torch.load(PARENT / 'best.pt', map_location='cpu', weights_only=False)
    state.update(phase='Loading consumed camera episodes for settled-height fitting', learning=dict(stage='loading'))
    seq=sequences()
    template_report=fit_banks(saved,seq)
    saved['source_template_fit']=template_report['families']['source']
    saved['cup_template_fit']=template_report['families']['cup']
    validation={e['name'] for e in seq if e['split']=='validation'}
    consensus,consensus_report=calibrate_localization(saved,seq,root=ROOT,validation=validation)
    saved['localization_consensus']=consensus
    saved['source_frame_normalization']='raw_and_cup_aligned_v1'
    bundle=build_data(saved,seq)
    empty_coverage=probe_empty_detector(saved,seq)
    loc=dict(template_fit=template_report,consensus=consensus_report,
        note='Consumed parent and expanded reference calibration windows; independent task evaluation remains reserved.')
    (ROOT/'localization-validation.json').write_text(json.dumps(loc,indent=2)+'\n')
    saved['pour_primitive'],motion_report=fit_motion(saved['pour_primitive'],seq)
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
    saved.update(model_version='visual_workspace_hierarchy_v18', settled_regressor=chosen['fit'],
        settled_validation=chosen['metrics'], parent_checkpoint_sha256=PARENT_SHA,
        method='Expanded-position gradient template calibration, supervised settled-height ridge regression and contextual joint-target behavioral cloning; visual servo, frozen neural weights with raw/cup-aligned source recognition and bounded feedback stopping')
    saved['thresholds'].update(full_settled_target=full_target, settled_geometry_tolerance=.10,
        settled_stability_std=.01, trim_increment=.005, trim_min_phase=95.,adaptive_trim=True,
        trim_extra_deg=.50,trim_hard_cap_deg=110.,trim_fast_rate_deg_s=35.,trim_creep_rate_deg_s=35.,trim_total_seconds=1.5,
        trim_seconds=0.,discrete_topup=True,max_pauses=10)
    saved['thresholds'].update(coarse_flow_min_level=.25,coarse_flow_limit_ml_s=18.)
    saved['thresholds'].update(approach_tracking_tolerance_rad=.02,approach_tracking_wait_s=.75)
    saved['pour_primitive']['slow_rate_deg_s']=5.
    saved['approach_clearance_m']=.04
    repair_plan=ROOT/'reference-repair-plan.json'
    saved['reference_repair_plan_sha256']=hashlib.sha256(repair_plan.read_bytes()).hexdigest() if repair_plan.exists() else None
    reference_acceptance=[dict(name=e['name'],reference_case=e['reference_case'],split=e['split'],
        accepted=e['accepted_reference'],gates=e['rejection_gates']) for e in seq if e['name'].startswith(PREFIX)]
    saved['training_device'] = dict(name=torch.cuda.get_device_name(0), visible_cuda_devices=1)
    saved['settled_training'] = dict(train_episodes=len(set(tr['episode'])), validation_episodes=len(set(va['episode'])),
        train_frames=len(y), validation_frames=len(vy), learned_parameters=len(chosen['fit']['coef']),
        objective='episode-balanced weighted squared error to final settled height + ridge penalty',
        frozen_visual_encoders=True, motion_training_episodes=len(motion_report['train_episodes']), motion_validation_episodes=len(motion_report['validation_episodes']), candidates=len(candidates), data_manifest_sha256=hashlib.sha256((ROOT / 'data-manifest.json').read_bytes()).hexdigest())
    saved['sources'] = {n: hashlib.sha256(Path('/workspace/src', n).read_bytes()).hexdigest() for n in RUNTIME_SOURCES}
    torch.save(saved, ROOT / 'best.partial.pt'); (ROOT / 'best.partial.pt').replace(ROOT / 'best.pt')
    summary = dict(checkpoint_sha256=hashlib.sha256((ROOT / 'best.pt').read_bytes()).hexdigest(),
        parent_checkpoint_sha256=PARENT_SHA, method=saved['method'], training=saved['settled_training'],
        device=saved['training_device'], chosen=dict(columns=chosen['fit']['columns'], ridge=chosen['fit']['ridge'], metrics=chosen['metrics']),
        thresholds=saved['thresholds'], localization=loc, localization_consensus=consensus_report, template_fit=template_report, motion=motion_report, empty_detector_coverage=empty_coverage,
        motion_rate_deg_s=dict(fast=saved['pour_primitive']['fast_rate_deg_s'],slow=saved['pour_primitive']['slow_rate_deg_s']), elapsed_s=time.monotonic() - started,
        source_frame_normalization=saved['source_frame_normalization'],
        approach_clearance_m=saved['approach_clearance_m'],approach_method='Raised Cartesian IK approach/retreat from RGB-localized target, measured home joints and fixed hardware geometry; unchanged collisions and drive limits',
        reference_repair_plan_sha256=saved['reference_repair_plan_sha256'],reference_acceptance=reference_acceptance,
        candidates=[dict(columns=c['fit']['columns'], ridge=c['fit']['ridge'], metrics=c['metrics'], score=c['score']) for c in candidates],
        unchanged_components=['cup CNN weights', 'source detector weights', 'empty confirmation gates', 'fluid and vessel physics', 'primary task scoring'],
        no_privileged_runtime_inputs=True, data_manifest_sha256=saved['settled_training']['data_manifest_sha256'], sources=saved['sources'])
    (ROOT / 'model-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    state.update(phase='Expanded-position fitting complete', learning=dict(stage='fit_complete', elapsed_s=summary['elapsed_s'], metrics=chosen['metrics']))
    del model, bundle, matrices, x, vx, y, vy; torch.cuda.empty_cache()
    return str(ROOT / 'best.pt')
