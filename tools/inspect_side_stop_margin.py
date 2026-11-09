"""Counterfactual stop-threshold timing on consumed traces; no new outcome claim."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/side-return-A';rows=[]
for part in ('development','test'):
    report=json.loads((ROOT/f'evidence/perception-A/{part}-r3-outcomes.json').read_text())
    for row in report['cases']:
        if row['expected_stop']!='cup_near_full':continue
        folder=ROOT/'data/material-A-perception'/Path(row['path']).relative_to('/workspace/outputs/fountain')
        d=json.loads((folder/'policy-decisions.json').read_text());s=json.loads((folder/'samples.json').read_text())
        variants=[]
        for target in (.93,.94,.95):
            pause=None;votes=0;selected=None;start=0.
            for i,x in enumerate(d):
                in_pause=x['fill_mode']=='settle' or (i>0 and d[i-1]['fill_mode']=='settle' and x['fill_mode']=='trim')
                if not in_pause:pause=None;votes=0;continue
                if pause!=x['pause_count']:pause=x['pause_count'];start=x['t'];votes=0
                if x['t']-start<1.2:continue
                recent=[a for a in d[:i+1] if a['t']>=start+.35][-6:]
                values=np.array([a['settled_prediction'] for a in recent]);other=np.median([a['surface_geometry_fraction'] for a in recent])
                good=len(values)==6 and np.std(values)<.01 and np.quantile(values,.25)>=target and abs(other-np.quantile(values,.25))<.1
                if good:
                    selected=dict(decision_t=x['t'],pause=pause,observed_next_surface=s[i]['cup_fill_fraction'],prediction_quartile=float(np.quantile(values,.25)))
                    break
                if x['stop_latched']:break
            variants.append(dict(target=target,first_eligible_frame=selected))
        rows.append(dict(name=row['name'],source_sha256=row['source_sha256'],decisions_sha256=hashlib.sha256((folder/'policy-decisions.json').read_bytes()).hexdigest(),variants=variants,actual_final_height=row['cup_surface_height_fraction']))
report=dict(cases=rows,scope='Consumed traces only. Finds the first recorded eligible visual gate only, not the later three-vote latch. The original controller leaves each rejected pause immediately, so later counterfactual votes are unobserved. Does not model altered fluid, observations or return after an earlier stop. No mass feature was recorded; all selected rows had the original valid cup view. Native development is required.',candidate_target=.94)
out=E/'stop-margin-first-eligible-v2.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(rows,indent=2))
