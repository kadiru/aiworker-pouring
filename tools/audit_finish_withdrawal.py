"""Measure actual over-cup dwell, set-down and release from audited scalar logs."""
import argparse,json
from pathlib import Path
import numpy as np
from finish_media_source import sha
PROJECT=Path(__file__).resolve().parents[1];DATA=PROJECT/'data/material-A-finish'

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--round',type=int);p.add_argument('--baseline-wide',action='store_true');a=p.parse_args();audit=read(a.manifest)
    data=PROJECT/'data/material-A-wide' if a.baseline_wide else DATA
    prefix='XW30_A_' if a.baseline_wide else 'FN30_A_'
    assert audit['native_checks_passed'] and not audit['pending_attempts']
    selected=[r for r in audit['files'] if r['name'].startswith(prefix+'dev_') and r['name'].endswith(f'_r{a.round}')] if a.round is not None else [r for r in audit['files'] if r['name'].startswith(prefix+'test_')]
    assert len(selected)==(8 if a.round is not None else 30)
    sidecars={r['relative_path']:r for r in audit['sidecar_files']};result=[]
    for r in selected:
        folder=(data/r['relative_path']).parent;loaded={}
        for name in ('samples.json','policy-decisions.json','parameters.json'):
            path=folder/name;identity=sidecars[str(path.relative_to(data))]
            assert path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256'];loaded[name]=read(path)
        samples,decisions,spec=loaded['samples.json'],loaded['policy-decisions.json'],loaded['parameters.json']
        assert len(samples)==len(decisions)
        stop=next((i for i,d in enumerate(decisions) if d['stop_latched']),None)
        returning=[i for i,d in enumerate(decisions) if d['mode']=='Returning upright'];contacts=[];dwell=longest=0.;speeds=[]
        for i in returning:
            if not i:continue
            s=samples[i];dt=s['t']-samples[i-1]['t'];xy=np.array(s['bottle_pose_xyz'][:2]);speed=float(np.linalg.norm(np.array(s['bottle_pose_xyz'])-samples[i-1]['bottle_pose_xyz'])/dt)
            angular_speed=abs(s['bottle_tilt_deg']-samples[i-1]['bottle_tilt_deg'])/dt
            over=np.linalg.norm(xy-np.array(spec['cup_world'][:2]))<.06
            still=over and abs(s['bottle_tilt_deg'])<8 and speed<.003 and angular_speed<5.
            dwell=dwell+dt if still else 0.;longest=max(longest,dwell)
        released=[i for i,d in enumerate(decisions) if d['mode'] in ('Retreating','Holding completed pose')]
        for i in released:
            found=[c for c in samples[i]['contacts'] if c.startswith('gripper_')]
            if found:contacts.append(dict(t=samples[i]['t'],contacts=found))
            if i:speeds.append(float(np.linalg.norm(np.array(samples[i]['bottle_pose_xyz'])-samples[i-1]['bottle_pose_xyz'])/(samples[i]['t']-samples[i-1]['t'])))
        modes=('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose')
        starts={mode:next((s['t'] for s,d in zip(samples,decisions) if d['mode']==mode),None) for mode in modes}
        ordered=all(t is not None for t in starts.values()) and all(starts[x]<starts[y] for x,y in zip(modes,modes[1:]))
        result.append(dict(name=r['name'],source_sha256=r['sha256'],stop_recorded=stop is not None,stage_starts_s=starts,
            terminal_sequence_recorded=ordered,return_sample_count=len(returning),post_release_sample_count=len(released),
            longest_sampled_upright_near_stationary_over_cup_s=longest if returning else None,
            post_release_gripper_contact_frames=len(contacts) if released else None,contacts=contacts,
            maximum_post_release_bottle_speed_m_s=max(speeds,default=None),final_tilt_deg=samples[-1]['bottle_tilt_deg'],
            final_base_height_m=samples[-1]['bottle_base_height_m'],final_outside_ml=samples[-1]['spilled_ml']+samples[-1]['airborne_ml']))
    out=dict(cases=result,source_manifest_sha256=sha(a.manifest),historical_baseline=a.baseline_wide,analysis_source_sha256=sha(Path(__file__)),
        definition='Over-cup dwell: bottle center within 6 cm horizontally of cup, tilt below 8 degrees, sampled 3D speed below 3 mm/s and tilt speed below 5 deg/s during the terminal return. Diagnostic only; strict task score is unchanged.',
        scope='20 Hz simulator logs, not a claim about unsampled contact, naturalness or physical hardware. Missing return or post-release observations have null measurements, not zero-dwell or zero-contact successes.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(result),maximum_dwell_s=max((r['longest_sampled_upright_near_stationary_over_cup_s'] for r in result if r['return_sample_count']),default=None),contact_cases=[r['name'] for r in result if r['contacts']],missing_terminal_sequences=[r['name'] for r in result if not r['terminal_sequence_recorded']]),indent=2))
if __name__=='__main__':main()
