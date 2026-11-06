"""Fit/check only the empty readout on audited references; no deployment."""
import argparse,hashlib,json,sys,time
from pathlib import Path

STUDY=Path('/workspace/learning/A-wide-20261105-v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);a=p.parse_args()
    assert not (a.candidate_dir/'component-result.json').exists()
    sys.path[:0]=[str(a.candidate_dir),'/workspace/src']
    import torch
    import pour_wide_study as study
    import pour_wide_train as train
    from pour_wide_source import probe
    from pour_wide_empty import fit_readout
    assert not (STUDY/'best.pt').exists() and not (STUDY/'final-selection.json').exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    assert hashlib.sha256((study.PARENT/'best.pt').read_bytes()).hexdigest()==study.PARENT_SHA
    train.ROOT=a.candidate_dir;torch.set_num_threads(2);started=time.monotonic()
    result=dict(device=torch.cuda.get_device_name(0),visible_cuda_devices=1,
        source_sha256={n:hashlib.sha256((a.candidate_dir/n).read_bytes()).hexdigest() for n in ('pour_wide_source.py','pour_wide_empty.py')},
        localized_crop_manifest_sha256=hashlib.sha256((STUDY/'data-manifest.json').read_bytes()).hexdigest(),
        scope='Empty-readout component fit on 24 training/eight validation references with fixed RGB-localized crops. No test data, candidate checkpoint, deployment or autonomous trial.')
    try:
        saved=torch.load(study.PARENT/'best.pt',map_location='cpu',weights_only=False)
        seq=train.sequences(saved)
        study.ROOT=a.candidate_dir
        localized={e['name']:e['cup_crop_xy'] for e in json.loads((STUDY/'data-manifest.json').read_text())
                   if e['name'].startswith(study.REFERENCE_PREFIX+'demo_')}
        with torch.inference_mode():
            fit=fit_readout(saved,seq,localized);saved['wide_empty_readout']=fit
            (a.candidate_dir/'empty-readout-summary.json').write_text(json.dumps(fit,indent=2)+'\n')
            coverage=probe(saved,seq,localized)
        (a.candidate_dir/'empty-detector-coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
        assert coverage['validation_negative_frames']>0
        assert not coverage['validation_empty_episodes_without_detection'] and not coverage['false_positive_frames']
        result.update(status='passed',fit=fit,coverage={k:v for k,v in coverage.items() if k!='episodes'})
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        result['elapsed_s']=time.monotonic()-started
        (a.candidate_dir/'component-result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
