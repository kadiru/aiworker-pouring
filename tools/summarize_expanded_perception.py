"""Compact, hash-bound results; full fits and recordings stay native."""
import argparse,datetime,hashlib,json
from pathlib import Path
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1'));p.add_argument('--output',required=True,type=Path);a=p.parse_args();root=a.root
    def read(n):return json.loads((root/n).read_text())
    rim=read('rim-d3.json');tracker=read('tracker-t2.json')
    summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rim={k:rim[k] for k in ('passed','parameters','steps','counts','cpu_gpu_max_pixels','checkpoint_sha256','total_seconds')},
        tracker=dict(passed=tracker['passed'],episodes=len(tracker['cases']),frames=len(tracker['rows']),
            passed_episodes=sum(r['passed'] for r in tracker['cases']),
            raw_initial_max_mm=max(r['raw_initial_max_mm'] for r in tracker['cases']),
            tracked_max_mm=max(r['tracked_max_mm'] for r in tracker['cases']),
            synthetic_rejections=sum(r['rejected'] for r in tracker['synthetic_challenges']),
            synthetic_total=len(tracker['synthetic_challenges']),cpu_gpu_max_pixels=tracker['cpu_gpu_max_pixels'],
            checkpoint_sha256=tracker['checkpoint_sha256'],source_report_sha256=sha(root/'tracker-t2.json')),
        fits={},cpu_checks=read('expanded-perception-cpu-checks.json'),
        active_controller_changed=False,autonomy_accepted=False)
    summary['rim'].update(selected=rim['selected']['row'],source_report_sha256=sha(root/'rim-d3.json'))
    if (root/'rim-d4.json').exists():
        r=read('rim-d4.json');summary['refined_midfill_rim']={k:r[k] for k in ('passed','parameters','steps','counts','cpu_gpu_max_pixels','checkpoint_sha256','total_seconds')}
        summary['refined_midfill_rim'].update(selected=r['selected']['row'],source_report_sha256=sha(root/'rim-d4.json'))
    if (root/'tracker-t3.json').exists():
        r=read('tracker-t3.json');summary['refined_midfill_tracker']=dict(passed=r['passed'],episodes=len(r['cases']),frames=len(r['rows']),
            passed_episodes=sum(v['passed'] for v in r['cases']),raw_initial_max_mm=max(v['raw_initial_max_mm'] for v in r['cases']),
            tracked_max_mm=max(v['tracked_max_mm'] for v in r['cases']),synthetic_rejections=sum(v['rejected'] for v in r['synthetic_challenges']),
            synthetic_total=len(r['synthetic_challenges']),cpu_gpu_max_pixels=r['cpu_gpu_max_pixels'],checkpoint_sha256=r['checkpoint_sha256'],source_report_sha256=sha(root/'tracker-t3.json'))
    for rnd in ('r8','r9','r10','r11','r12'):
        if not (root/f'pair-{rnd}.json').exists():continue
        pair=read(f'pair-{rnd}.json');item=dict(pair,source_report_sha256=sha(root/f'pair-{rnd}.json'),branches={})
        for branch in ('rgb','color'):
            name=f'pair-{rnd}-{branch}.json';r=read(name)
            item['branches'][branch]=dict(passed=r['passed'],selected=r['selected'],cpu_gpu_difference=r['cpu_gpu_difference'],
                inference_path_difference=r['inference_path_difference'],absent_color_rejected=r['absent_color_rejected'],
                checkpoint_sha256=r['checkpoint_sha256'],source_report_sha256=sha(root/name))
        summary['fits'][rnd]=item
    for name,key in [('level-cache-v4.json','relocalized_cache'),('level-cache-v5.json','midfill_cache'),
                     ('level-cache-v6.json','refined_midfill_cache'),('midfill-tracker-admission.json','midfill_admission'),('train-r3-manual-review.json','midfill_review')]:
        if (root/name).exists():
            r=read(name);summary[key]={k:v for k,v in r.items() if k not in ('admissions','cases')};summary[key]['source_report_sha256']=sha(root/name)
    native_outputs=root.parents[1]/'outputs/fountain'
    started=list(native_outputs.glob('*/*SD_A_motion_holdout_*/parameters.json'))
    summary['reserved_started']=len(started)
    if (root/'moving-holdout-evaluation.json').exists():
        r=read('moving-holdout-evaluation.json')
        summary['reserved_result']=dict(passed=r['passed'],passed_cases=r['passed_cases'],cases=[{k:v for k,v in row.items() if k not in ('frames','initial_raw_errors_mm')} for row in r['cases']],source_report_sha256=sha(root/'moving-holdout-evaluation.json'))
    with a.output.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(a.output),tracker_passed=summary['tracker']['passed'],fits={k:v['passed'] for k,v in summary['fits'].items()},reserved_started=len(started))),flush=True)
if __name__=='__main__':main()
