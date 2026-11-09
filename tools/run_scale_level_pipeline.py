"""One guarded level repair; frozen rim, all gates before once-only static tests."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, '/tmp')
import watch_wide_collection as w
ROOT = w.HOST_ROOT/'learning/A-scale-20261109-v1'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v):
    with p.open('x') as f:
        json.dump(v, f, indent=2); f.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r7','r8','r9','r10'),default='r7');args=parser.parse_args()
    m=read(ROOT/'candidate.json'); plan=read(ROOT/'plan.json')
    protocol=read(ROOT/f'level-refit-{args.round}-plan.json')
    assert plan['version'] in (5,6) and protocol['round']==args.round
    assert all(sha(ROOT/n)==v for n,v in protocol['frozen_locator'].items())
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    review=read(ROOT/m['collection_review']); audit=read(ROOT/m['collection_audit'])
    assert review['accepted'] and review['audit_sha256']==sha(ROOT/m['collection_audit'])
    assert audit['complete'] and audit['labels_accepted']==len(plan['collect'])
    assert set(review['cases']) | set(review['previously_reviewed']) == {s['name'] for s in plan['collect']}
    assert not set(review['cases']) & set(review['previously_reviewed'])
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    output=ROOT/f'level-pipeline-{args.round}'; output.mkdir()
    write(output/'runtime-before.json',w.isolation())
    phases=[]
    def run(name, command):
        with (output/(name+'.log')).open('x') as f:
            result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        phases.append(dict(stage=name,exit_code=result.returncode,log_sha256=sha(output/(name+'.log'))))
        print(json.dumps(phases[-1]),flush=True)
        if result.returncode:
            finish(name+'_execution_failed')
            raise RuntimeError(name+' failed; no dependent stage was launched')
    def native(helper, *args):
        return ['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1',w.CID,
                'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-scale-20261109-v1/'+helper,*args]
    def finish(stage):
        h=w.health();runtime=w.isolation()
        assert not h['queue'] and not h['recording']['active'] and not h.get('error')
        assert sha(ROOT/'best.pt')==m['checkpoint_sha256'] and sha(ROOT/'plan.json')==m['plan_sha256']
        assert all(sha(ROOT/n)==v for n,v in protocol['frozen_locator'].items())
        assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in m['runtime_sources'].items())
        assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==d for p,d in m['protected_parent_files'].items())
        started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
        tested=started & {s['name'] for s in plan['holdout']}
        if stage in ('level_gate_failed','level_execution_failed','select_execution_failed'):assert not tested
        reports={n:sha(ROOT/n) for n in ('cup-neural-localization.json','static-level-fit.json',
                  'level-localization-check.json','component-selection.json','holdout-r0-audit.json',
                  'static-holdout-evaluation.json') if (ROOT/n).exists()}
        outcomes={key:(read(ROOT/name)['passed'] if name in reports else None) for key,name in
                  (('locator_passed','cup-neural-localization.json'),('level_passed','static-level-fit.json'),
                   ('independent_static_passed','static-holdout-evaluation.json'))}
        report=dict(stage=stage,phases=phases,plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],
            reports=reports,reserved_scenes_started=len(tested),active_controller_unchanged=True,
            autonomous_size_accepted=False,queue_empty=True,recording_inactive=True,runtime=runtime,
            helper_sha256=sha(Path(__file__)),**outcomes,scope='Static known-size supervised perception only; no autonomous pouring task result.')
        write(output/'completion.json',report);print(json.dumps(dict(stage=stage,reserved_scenes_started=len(tested))),flush=True)
        current=read(ROOT/'candidate.json');current.update(stage=stage,level_pipeline_completion_sha256=sha(output/'completion.json'),
            completed_level_fits=m.get('completed_level_fits',1)+int(any(p['stage']=='level' and p['exit_code']==0 for p in phases)),
            accepted_locator=outcomes['locator_passed'] is True,static_level_accepted=outcomes['level_passed'] is True)
        temp=ROOT/'candidate.tmp';temp.write_text(json.dumps(current,indent=2)+'\n');temp.replace(ROOT/'candidate.json')
    if not read(ROOT/'cup-neural-localization.json')['passed']:
        finish('localization_gate_failed');return
    run('level',['python3',str(ROOT/'run_scale_locator_fit.py'),'--component','level','--round',args.round])
    if not read(ROOT/'static-level-fit.json')['passed']:
        finish('level_gate_failed');return
    run('select',native('select_scale_static_components.py'))
    run('holdout_collection',['python3','/tmp/aiworker-scale-stage-transfer/watch_scale_stage.py','--part','holdout','--round','r0'])
    run('holdout_audit',native('check_scale_stage.py','--part','holdout','--round','r0'))
    run('holdout_score',native('evaluate_scale_static_holdout.py'))
    finish('independent_static_evaluation_complete')


if __name__=='__main__':main()
