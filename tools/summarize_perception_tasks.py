"""Independent strict task scoring and terminal-motion checks, without fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def withdrawal(samples,decisions,spec):
    assert len(samples)==len(decisions)
    modes=('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose')
    starts={m:next((s['t'] for s,d in zip(samples,decisions) if d['mode']==m and d['stop_latched']),None) for m in modes}
    ordered=all(t is not None for t in starts.values()) and all(starts[a]<starts[b] for a,b in zip(modes,modes[1:]))
    dwell=longest=0.;returning=0;released=0;contact_frames=0
    for i,(s,d) in enumerate(zip(samples,decisions)):
        if not d['stop_latched']:continue
        if d['mode']=='Returning upright' and i:
            returning+=1;previous=samples[i-1];dt=s['t']-previous['t'];assert dt>0
            speed=np.linalg.norm(np.array(s['bottle_pose_xyz'])-previous['bottle_pose_xyz'])/dt
            angular_speed=abs(s['bottle_tilt_deg']-previous['bottle_tilt_deg'])/dt
            over=np.linalg.norm(np.array(s['bottle_pose_xyz'][:2])-spec['cup_world'][:2])<.06
            still=over and abs(s['bottle_tilt_deg'])<8 and speed<.003 and angular_speed<5
            dwell=dwell+dt if still else 0.;longest=max(longest,dwell)
        if d['mode'] in modes[-2:]:
            released+=1;contact_frames+=any(c.startswith('gripper_') for c in s['contacts'])
    result=dict(stage_starts_s=starts,ordered_terminal_sequence=ordered,return_samples=returning,post_release_samples=released,
        longest_sampled_upright_stationary_over_cup_s=longest if returning else None,
        post_release_contact_frames=contact_frames if released else None,final_tilt_deg=samples[-1]['bottle_tilt_deg'],
        final_base_height_m=samples[-1]['bottle_base_height_m'])
    result['passed']=bool(ordered and returning and released and longest<=.35 and contact_frames==0
        and abs(result['final_tilt_deg'])<8 and abs(result['final_base_height_m'])<=.005)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('development','test'),required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');checkpoint=sha(ROOT/'best.pt')
    assert checkpoint==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    audit=read(ROOT/'recording-audit.json');index={r['name']:r for r in audit['records']};records=[]
    for spec in plan[a.part]:
        name=spec['name'];record=index.get(name)
        if record is None:
            records.append(dict(name=name,status='incomplete' if name in audit['started_incomplete'] else 'pending'));continue
        path=Path(record['path']);assert sha(path)==record['sha256'];actual=read(path.parent/'parameters.json')
        assert all(actual[k]==v for k,v in spec.items())
        for n,d in record['sidecars'].items():assert sha(path.parent/n)==d
        row=completed_row(path.parent,checkpoint);row.update(strict_pass=strict_success(row),failure_reasons=reasons(row),source_sha256=record['sha256'])
        assert row['strict_pass']==(not row['failure_reasons'])
        decisions=read(path.parent/'policy-decisions.json');samples=read(path.parent/'samples.json')
        row['withdrawal']=withdrawal(samples,decisions,spec)
        assert all(d.get('cup_level_camera')=='cup_detail' and d.get('source_presence_model')=='wrist_presence_cnn' for d in decisions)
        records.append(row)
    done=[r for r in records if r['status']=='complete']
    counts=dict(planned=len(records),completed=len(done),strict_passes=sum(r['strict_pass'] for r in done),
        zero_measured_loss=sum(r['zero_spill'] for r in done),completed_returns=sum(r['withdrawal']['passed'] for r in done))
    if a.require_complete:assert len(done)==len(records)
    report=dict(status='complete' if len(done)==len(records) else 'incomplete',part=a.part,counts=counts,cases=records,
        checkpoint_sha256=checkpoint,plan_sha256=manifest['plan_sha256'],runtime_sources=manifest['runtime_sources'],
        audit_sha256=sha(ROOT/'recording-audit.json'),helper_sha256=sha(Path(__file__)),
        primary_definition='Original strict score: verified physical/visual grasp, >=80% bilateral contact during tipping, >=1 nominal mL transfer, correct stop, >=90% final sampled rim height or source <=max(0.25 nominal mL, 1% initial), completed return/release, no gates/fallback, zero final outside and ever-near-worktop loss.',
        scope='Native Isaac Sim RGB/joint controller evaluation at 20 Hz; no simulator truth in control. Declared baseline cup/bottle sizes and new small position jitters of prior layout families. Not novel-layout or physical-hardware validation. All attempts retained; a no-pour zero-loss case cannot pass.')
    (ROOT/(a.part+'-outcomes.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
