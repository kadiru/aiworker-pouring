"""Conservative bottle/cup envelope screen during early tilt, no wet trials."""
import os,sys
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/learning/A-wide-20261105-v1/staged','/workspace']
import check_workspace_clearance
from preflight_wide_study import vessel
from pour_wide_study import ROOT,plan
from bottle_pour import pose,ease
from pour_wide_geometry import calibrated_pose
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import numpy as np,json,argparse

p=argparse.ArgumentParser();p.add_argument('--candidate',action='store_true');p.add_argument('--segmented',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()

cup,cup_eq=vessel(np.zeros(3),'cup');body,_=vessel(np.zeros(3),'bottle');rows=[]
# Keep the neck's concavity: a single whole-bottle convex hull fills the
# space outside its narrow neck. Each consecutive profile pair is convex.
pieces=[body[i*48:(i+2)*48] for i in range(3)] if a.segmented else [body]
for s in sum((plan()[k] for k in ('demonstrations','development','test')),[]):
    hits=[]
    for centered in (False,True) if a.candidate else (False,):
        for angle in np.arange(0.,166.,2.):
            if a.candidate:base,rotation=calibrated_pose(angle,1.,s['source_xy'],s['mouth_z'],centered=centered)
            else:
                base,rotation=pose(angle,1.,s['source_xy'],s['mouth_z']);base[2]+=.018*(1-ease(angle/60.))
            for piece_index,body_piece in enumerate(pieces):
                points=body_piece@rotation.T+base
                if not (np.minimum(points.max(0),cup.max(0))>np.maximum(points.min(0),cup.min(0))).all():continue
                eq=ConvexHull(points).equations
                result=linprog(np.zeros(3),A_ub=np.vstack([cup_eq[:,:3],eq[:,:3]]),b_ub=-np.r_[cup_eq[:,3],eq[:,3]],bounds=[(None,None)]*3,method='highs')
                assert result.status in (0,2)
                if result.success:hits.append(dict(angle_deg=float(angle),centered_return=centered,piece=piece_index))
    rows.append(dict(name=s['name'],hits=hits))
(ROOT/a.output).write_text(json.dumps(dict(candidate=a.candidate,segmented_bottle=a.segmented,cases=rows,
 scope='Conservative full outer envelopes, wall thickness and 2mm margin, along commanded reference poses. Native tracking and hollow-vessel contacts are not modeled; positives are geometry flags, not asserted native collisions.'),indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),flagged_cases=sum(bool(r['hits']) for r in rows),example=next((r for r in rows if r['hits']),None))))
