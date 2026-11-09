"""Declare diagnostic reuse, six development fixtures and thirty fresh holdouts."""
import copy,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/side-return-A'
P='/workspace/learning/A-side-return-20261108-v1'

def main():
    E.mkdir(exist_ok=True)
    out=E/'plan-initial.json';assert not out.exists()
    parent=json.loads((ROOT/'evidence/perception-A/plan-r3-exact-quantiles.json').read_text())
    rng=random.Random(202611080731)
    def spec(s,name,split):
        s=copy.deepcopy(s);s.update(name=name,study_split=split,checkpoint=P+'/best.pt',
            save_particle_trajectory=True,trajectory_stride=1,trajectory_start_s=18.,record_particle_diagnostics=True)
        return s
    diagnostics=[spec(parent['test'][i],f'SR_A_diagnostic_00_{i:02}','diagnostic') for i in (0,4,2,3)]
    for s,i in zip(diagnostics,(0,4,2,3)):
        s.update(consumed_parent_case=parent['test'][i]['name'],evidence_scope='Consumed development diagnostic; not fresh evaluation')
    # Same baseline vessel geometry and previously declared layout families.
    # Fresh independent position/inventory realizations, not novel-workspace claims.
    families=parent['development']+parent['test']
    def fresh(i,split):
        old=families[i%len(families)];s=spec(old,f'SR_A_{split}_r0_{i:02}',split)
        s['parent_fixture']=old['name']
        s['cup_world'][:2]=[x+rng.uniform(-.003,.003) for x in old['cup_world'][:2]]
        s['source_xy']=[x+rng.uniform(-.002,.002) for x in old['source_xy']]
        full=(i%2==0)
        s.update(initial_ml=float(rng.randint(88,98) if full else rng.randint(25,38)),
            cup_initial_ml=float(rng.randint(4,10)),expected_stop='cup_near_full' if full else 'bottle_empty')
        # Preserve the runtime stop vocabulary from the established full fixtures.
        if full:s['expected_stop']=next(x['expected_stop'] for x in families if x['initial_ml']>80)
        return s
    plan=dict(version=1,diagnostic=diagnostics,development=[fresh(i,'development') for i in range(6)],
        test=[fresh(i,'test') for i in range(30)],seed=202611080731,
        scope='Baseline size, material A; new jitters/inventories within prior families. Robot-right withdrawal = negative world Y; left fallback.',
        task_score=parent['strict_task_metric'],
        development_gate='Six strict task passes, zero final/table loss, complete side placement, reviewed videos before 30 tests.',
        test_rule='Once-only after immutable selection; no fitting/tuning or replacement cases during evaluation.',
        method='Retained supervised RGB perception and motion BC; programmed visual stop supervisor and IK terminal trajectory. No RL or hardware validation.')
    out.write_text(json.dumps(plan,indent=2)+'\n');print(out)
if __name__=='__main__':main()
