"""Diagnose episode weighting on an immutable, audited component cache."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.candidate_dir
    output = root / 'weighting-result.json'
    assert not output.exists()
    sys.path[:0] = [str(root), '/workspace/src']
    import torch
    from pour_guard_models import CupEstimator
    from pour_settled_model import features, predict
    assert torch.cuda.device_count() == 1 and '4070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    started = time.monotonic()
    bundle = torch.load(root / 'wide-cache.pt', map_location='cpu', weights_only=False)
    saved = torch.load('/workspace/learning/A-wide-20261105-v1/best.pt', map_location='cpu', weights_only=False)
    model = CupEstimator().cuda().eval()
    model.load_state_dict(saved['cup_model'])
    matrices = {}
    with torch.inference_mode():
        for split, d in bundle['data'].items():
            parts = []
            for start in range(0, len(d['target']), 128):
                image = d['image'][start:start+128].cuda()
                cnn = (model(image) + saved['cup_bias']).clamp(0, 1)
                parts.append(features(image, cnn, d['cup_xy'][start:start+128].cuda()))
            matrices[split] = torch.cat(parts).double()
        tr, va = bundle['data']['train'], bundle['data']['validation']
        x, vx = matrices['train'], matrices['validation']
        y, vy = tr['target'].cuda().double(), va['target'].cuda().double()
        counts = Counter(tr['episode'])
        dev = torch.tensor([n.startswith('XW30_A_dev') for n in tr['episode']], device='cuda')
        wide = torch.tensor([n.startswith('T30_A_demo') for n in tr['episode']], device='cuda')
        vwide = torch.tensor([n.startswith('T30_A_demo') for n in va['episode']], device='cuda')
        rows = []
        for columns in (list(range(21)), list(range(21))+list(range(21,25))+list(range(39,43))+list(range(57,60)), list(range(60))):
            a = x[:, columns]
            mean, scale = a.mean(0), a.std(0).clamp(min=.001)
            mean[0], scale[0] = 0., 1.
            a = (a-mean)/scale
            for wwide in (1., 4., 16.):
                for wdev in (1., 4., 16., 64.):
                    weights = y.new_tensor([1./counts[n] for n in tr['episode']])*torch.where(y>.8,3.,1.)
                    weights *= torch.where(wide, wwide, 1.)*torch.where(dev, wdev, 1.)
                    weights /= weights.mean()
                    for ridge in (.001, .01, .1, 1.):
                        penalty = torch.eye(len(columns),device='cuda',dtype=torch.float64)*ridge
                        penalty[0,0] = 1e-9
                        coef = torch.linalg.solve(a.T@(a*weights[:,None])+penalty,a.T@(y*weights))
                        fitted = dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),coef=coef.cpu().tolist(),ridge=ridge)
                        error, de = predict(vx,fitted)-vy, predict(x,fitted)-y
                        near, dn = vy>.8, dev&(y>.8)
                        metrics = dict(val_mae=float(error.abs().mean()),val_near_mae=float(error[near].abs().mean()),
                            val_near_p95=float(torch.quantile(error[near].abs(),.95)), val_near_over=float(torch.quantile(error[near],.95)),
                            wide_mae=float(error[vwide].abs().mean()),wide_near_max=float(error[vwide&near].abs().max()),
                            wide_near_over=float(torch.quantile(error[vwide&near],.95)),dev_max=float(de[dev].abs().max()),
                            dev_near_max=float(de[dn].abs().max()),dev_near_over=float(de[dn].max()))
                        eligible=metrics['dev_max']<.04 and metrics['dev_near_max']<.02 and metrics['wide_near_max']<.04
                        per_episode={}
                        for name in sorted(set(tr['episode'])):
                            if not name.startswith('XW30_A_dev'):continue
                            mask=torch.tensor([n==name for n in tr['episode']],device='cuda')
                            per_episode[name]=dict(target=float(y[mask][0]),min_error=float(de[mask].min()),max_error=float(de[mask].max()))
                        rows.append(dict(parameters=len(columns),ridge=ridge,wide_weight=wwide,development_weight=wdev,
                            eligible=eligible,score=metrics['val_near_mae']+.5*metrics['val_near_p95']+metrics['wide_mae'],metrics=metrics,per_episode=per_episode))
    rows.sort(key=lambda r:(not r['eligible'],r['score']))
    result=dict(device=torch.cuda.get_device_name(0),elapsed_s=time.monotonic()-started,
        cache_sha256=hashlib.sha256((root/'wide-cache.pt').read_bytes()).hexdigest(),candidates=rows,
        scope='Diagnostic only: audited reference and consumed development data, no runtime/model mutation or reserved tests.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(eligible=sum(r['eligible'] for r in rows),best=rows[0],elapsed_s=result['elapsed_s'])),flush=True)


if __name__ == '__main__':
    main()
