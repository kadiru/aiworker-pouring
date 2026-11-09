"""Read-only closed-case scores while the finite batch/audit is still running."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_side_cases import ROOT,read,sha,w,completed_row,strict_success,reasons
from side_recording_ready import committed_recording_paths
sys.path.insert(0,str(ROOT))
from summarize_side_tasks import withdrawal
from pour_side_metrics import side_metrics

p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);a=p.parse_args()
m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json')
assert sha(ROOT/'best.pt')==m['checkpoint_sha256'] and sha(ROOT/'plan.json')==m['plan_sha256']
assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in m['runtime_sources'].items())
if a.part=='test':
    frozen=read(ROOT/'final-selection.json')
    for k in ('checkpoint_sha256','plan_sha256','runtime_sources'):assert m[k]==frozen[k]
rows=[]
for spec in plan[a.part]:
    paths=list((w.HOST_ROOT/'outputs/fountain').glob('*/*'+spec['name']+'-*/metrics.json'))
    if not paths:continue
    assert len(paths)==1;folder=paths[0].parent
    native=str(Path('/workspace')/folder.relative_to(w.HOST_ROOT))
    if native not in committed_recording_paths(folder.parent):continue
    actual=read(folder/'parameters.json');assert all(actual[k]==v for k,v in spec.items())
    samples=read(folder/'samples.json');decisions=read(folder/'policy-decisions.json')
    original=completed_row(folder,m['checkpoint_sha256']);side=side_metrics(samples,decisions,spec);terminal=withdrawal(samples,decisions,spec)
    margin=max(s['cup_fill_fraction'] for s in samples)<=.98;strict=strict_success(original)
    failures=reasons(original)+(['side_return_unconfirmed'] if not side['passed'] else [])+(['withdrawal_unconfirmed'] if not terminal['passed'] else [])+(['insufficient_stopping_margin'] if not margin else [])
    rows.append(dict(name=spec['name'],strict_pass=strict,zero_spill=original['zero_spill'],height_fraction=original['cup_surface_height_fraction'],source_ml=original['source_remaining_ml'],side=side,withdrawal=terminal,stopping_margin_passed=margin,accepted=strict and side['passed'] and terminal['passed'] and margin,acceptance_failures=failures,sidecar_sha256={n:sha(folder/n) for n in ('parameters.json','metrics.json','samples.json','policy-decisions.json')}))
print(json.dumps(dict(part=a.part,planned=len(plan[a.part]),completed=len(rows),accepted=sum(r['accepted'] for r in rows),cases=rows,checkpoint_sha256=m['checkpoint_sha256'],plan_sha256=m['plan_sha256'],helper_sha256=sha(Path(__file__)),scope='Independent committed-case scores during the batch. Full recording audit and visual review remain pending. No queue, replay, fitting, source or parameter changes.'),indent=2))
