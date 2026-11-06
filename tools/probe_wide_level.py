"""Fit the stable-level component on explicit consumed data, without deployment."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);a=p.parse_args()
    root=Path('/workspace/learning/A-wide-20261105-v1');assert not (root/'final-selection.json').exists()
    output=a.candidate_dir/'component-result.json';assert not output.exists()
    for name in ('frozen-plan.json','development-recording-integrity.json'):shutil.copy2(root/name,a.candidate_dir/name)
    sys.path[:0]=[str(a.candidate_dir),'/workspace/src']
    import torch
    import pour_wide_study as study
    import pour_wide_train as train
    from pour_guard_vision import make_bank,project
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0);torch.set_num_threads(2)
    study.ROOT=a.candidate_dir;train.ROOT=a.candidate_dir;started=time.monotonic()
    result=dict(device=torch.cuda.get_device_name(0),source_sha256={n:hashlib.sha256((a.candidate_dir/n).read_bytes()).hexdigest() for n in ('pour_wide_train.py','pour_guard_vision.py')},scope='Component fit on reference data plus explicitly consumed development r0 stable-level images. Original validation episodes retained; no reserved tests or deployed checkpoint changes.')
    try:
        saved=torch.load(study.PARENT/'best.pt',map_location='cpu',weights_only=False)
        current=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
        with torch.inference_mode():
            seq=train.sequences(saved);dev=train.development_calibration(saved);examples=[e for e in seq if e['split']=='train']
            bank=make_bank([e['images'][0] for e in examples],[project(e['cup_xyz']+[0,0,.038],saved['calibration']) for e in examples],'cup',aggregation='max',absolute_gradients=True,anchor_radius_px=32.,gradient_clip=.05)
            for key in ('bounds','initial_region_uv','initial_workspace_xy_m'):bank[key]=current['cup_bank'][key]
            saved['cup_bank']=bank;bundle=train.level_data(saved,seq,dev);chosen,training=train.fit_level(saved,bundle)
        torch.save(saved['settled_regressor'],a.candidate_dir/'level-candidate.pt')
        result.update(status='passed',chosen={k:v for k,v in chosen.items() if k!='fit'},training=training,full_target=saved['thresholds']['full_settled_target'])
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        candidates=a.candidate_dir/'level-candidates.json'
        if candidates.exists():result['candidates']=[{k:v for k,v in r.items() if k!='fit'} for r in json.loads(candidates.read_text())]
        result['elapsed_s']=time.monotonic()-started;output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='candidates'}),flush=True)


if __name__=='__main__':main()
