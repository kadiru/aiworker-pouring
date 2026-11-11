"""Fixed equal-weight development screen; numerical/native acceptance remains pending."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1')
def read(n):return json.loads((ROOT/n).read_text())
def sha(n):return hashlib.sha256((ROOT/n).read_bytes()).hexdigest()

def metrics(rows,field):
    errors=[r[field]-r['target'] for r in rows];near=[e for r,e in zip(rows,errors) if r['target']>.85]
    return dict(mae=sum(abs(e) for e in errors)/len(errors),near_max=max(map(abs,near),default=0),near_overestimate=max(near,default=0))

def main():
    assert not (ROOT/'component-selection.json').exists()
    assert not list((ROOT.parents[1]/'outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    files=['protocol.json','pair-r12-rgb.json','pair-r12-rgb.pt','pair-r14-rgb.json','pair-r14-rgb.pt','pair-r14-color.json','pair-r14-color.pt','screen_rgb_ensemble.py']
    plan=dict(weights=[.5,.5],rgb_members=['pair-r12-rgb.pt','pair-r14-rgb.pt'],color='pair-r14-color.pt',
              frozen_files={n:sha(n) for n in files},
              scope='Fixed equal-weight RGB ensemble chosen before this aggregate screen. Complementary direct-regression and relative-landmark representations. Color weights/predictions are never inputs to the RGB ensemble. Existing development outputs only; full-precision CPU/GPU re-evaluation is still required.')
    with (ROOT/'ensemble-r17-plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    reports=[read(n) for n in ('pair-r12-rgb.json','pair-r14-rgb.json','pair-r14-color.json')]
    def key(r):return r['split'],r['name'],r['index']
    indexed=[{key(r):r for r in p['rows']} for p in reports]
    assert len(indexed[0])==4113 and all(d.keys()==indexed[0].keys() for d in indexed)
    rows=[]
    for k,a in indexed[0].items():
        b,c=indexed[1][k],indexed[2][k]
        assert abs(a['target']-b['target'])<1e-7 and abs(a['target']-c['target'])<1e-7
        rows.append(dict(a,rgb=.5*a['prediction']+.5*b['prediction'],color=c['prediction']))
    g=read('protocol.json')['gates'];scores={};episodes={};checks=[]
    for branch in ('rgb','color'):
        scores[branch]={};episodes[branch]={}
        for split in ('train','validation'):
            subset=[r for r in rows if r['split']==split];groups=defaultdict(list)
            for r in subset:
                size=(r['cup_height_scale'],r['cup_diameter_scale'])
                groups[('all',size)].append(r);groups[('moving' if r.get('moving') else 'static',size)].append(r)
            scores[branch][split]={str(k):metrics(v,branch) for k,v in groups.items()}
            episodes[branch][split]={n:metrics([r for r in subset if r['name']==n],branch)['mae'] for n in sorted({r['name'] for r in subset})}
            for m in scores[branch][split].values():checks.append(m['mae']<g['level_validation_mae'] and m['near_max']<g['level_validation_near_max_error'] and m['near_overestimate']<g['level_validation_near_overestimate'])
            checks.append(max(episodes[branch][split].values())<g['level_validation_mae'])
    disagreement=max(abs(r['rgb']-r['color']) for r in rows);checks.append(disagreement<g['crosscheck_max_disagreement'])
    result=dict(development_accuracy_screen_passed=all(checks),accepted=False,native_numerical_checks_pending=True,
                scores=scores,per_episode_mae=episodes,max_disagreement=disagreement,plan_sha256=sha('ensemble-r17-plan.json'),reserved_started=False)
    with (ROOT/'ensemble-r17-screen.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(screen_passed=all(checks),worst_episode_pp={b:max(v for s in e.values() for v in s.values())*100 for b,e in episodes.items()},max_disagreement_pp=100*disagreement,numerical_checks_pending=True)),flush=True)
if __name__=='__main__':main()
