"""GPU localization component check on references, without a deployed model."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path

STUDY=Path('/workspace/learning/A-wide-20261105-v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);a=p.parse_args()
    assert not (a.candidate_dir/'component-result.json').exists()
    sys.path[:0]=[str(a.candidate_dir),'/workspace/src']
    import numpy as np
    import torch
    import pour_wide_train as train
    import pour_guard_vision as vision
    from pour_wide_study import PARENT,PARENT_SHA
    assert not (STUDY/'best.pt').exists() and not (STUDY/'final-selection.json').exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    assert hashlib.sha256((PARENT/'best.pt').read_bytes()).hexdigest()==PARENT_SHA
    torch.set_num_threads(2);train.ROOT=a.candidate_dir;started=time.monotonic()
    result=dict(device=torch.cuda.get_device_name(0),visible_cuda_devices=1,
        source_sha256={n:hashlib.sha256((a.candidate_dir/n).read_bytes()).hexdigest()
            for n in ('pour_guard_vision.py','pour_wide_train.py')},
        scope='Component fitting/checks on 24 training and eight validation references only. No test data, candidate checkpoint, runtime deployment or autonomous trial.')
    try:
        saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
        legacy_bank=saved['source_bank']
        seq=train.sequences(saved)
        with torch.inference_mode():
            localization=train.fit_localization(saved,seq)
            result['localization']={k:dict(max_error_mm=v['max_error_mm'],margin=v['margin'],
                workspace_xy_m=v['initial_workspace_xy_m']) for k,v in localization['families'].items()}
            # Optional-region support must preserve old checkpoint matching.
            spec=importlib.util.spec_from_file_location('legacy_guard_vision','/workspace/src/pour_guard_vision.py')
            legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
            bank=legacy_bank
            for e in seq[:8]:
                uv,score=vision.locate(e['images'][0],bank)
                old_uv,old_score=legacy.locate(e['images'][0],bank)
                assert np.allclose(uv,old_uv,atol=1e-6,rtol=0) and abs(score-old_score)<1e-6
            result['legacy_matching_checks']=8
            tracking=train.fit_tracking(saved,seq)
        result.update(status='passed',tracking=dict(min_votes=tracking['min_votes'],max_error_px=tracking['max_error_px'],
                validation_max_error_px=tracking['validation_max_error_px']))
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        result['elapsed_s']=time.monotonic()-started
        (a.candidate_dir/'component-result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
