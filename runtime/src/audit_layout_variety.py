"""Audit all twelve unchanged-policy trials; preserve failures and native media."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from pour_layout_variety import ROOT, PREFIX, PARENT, CHECKPOINT_SHA, plan
from pour_robust_report import completed_row
from validate_pour_recording import validate, h5py, decode

OUTPUTS=Path('/workspace/outputs/fountain')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def item(path): return dict(relative_path=str(path.relative_to(OUTPUTS)),bytes=path.stat().st_size,sha256=sha(path))

def failure_reasons(row):
    reasons=[]
    if row['stop_reason']=='localization_unconfirmed': reasons.append('Object localization rejected; no confirmed grasp')
    elif row['stop_reason']=='approach_unconfirmed': reasons.append('Arm did not reach the grasp pose')
    elif not row['visual_grasp_confirmed'] or not row['grasp_verified']: reasons.append('Grasp was not confirmed')
    if row['delivered_to_cup_ml']<1.: reasons.append('No meaningful liquid transfer')
    if row['final_outside_ml']>0 or row['ever_below_worktop_ml']>0: reasons.append('Liquid escaped the bottle/cup')
    if row['grasp_verified'] and row['sustained_grasp_fraction']<.8: reasons.append('Two-finger grip was not sustained')
    if row['expected_stop']=='cup_near_full' and row['cup_surface_height_fraction']<.90 and row['delivered_to_cup_ml']>=1.: reasons.append('Cup finished below 90% rim height')
    if row['grasp_verified'] and not row['completed_return_release_sequence']: reasons.append('Stop and return sequence did not complete')
    if row['grasp_verified'] and row['stop_reason']!=row['expected_stop']: reasons.append('Incorrect or missing stop decision')
    return reasons

def main():
    global ROOT,PREFIX,plan
    parser=argparse.ArgumentParser();parser.add_argument('--extra',action='store_true');parser.add_argument('--source-root',type=Path,default=Path('/workspace/src'));args=parser.parse_args()
    if args.extra:
        from pour_layout_extra import ROOT as extra_root,PREFIX as extra_prefix,plan as extra_plan
        ROOT,PREFIX,plan=extra_root,extra_prefix,extra_plan
    selection=json.loads((ROOT/'selection.json').read_text());p=plan()
    assert sha(PARENT/'best.pt')==CHECKPOINT_SHA
    assert sha(ROOT/'frozen-plan.json')==selection['scene_plan_sha256']
    for name,digest in selection['runtime_sources'].items():assert sha(args.source_root/name)==digest,name
    folders={}
    for path in OUTPUTS.glob(f'*/*{PREFIX}*/parameters.json'):
        name=json.loads(path.read_text())['name'];assert name not in folders;folders[name]=path.parent
    assert set(folders)=={s['name'] for s in p['test']} and len(folders)==len(p['test'])
    geometry_path=ROOT/'posthoc-complete-hand-geometry.json'
    geometry={r['name']:r for r in json.loads(geometry_path.read_text())['cases']} if geometry_path.exists() else {}
    rows=[];files=[];sidecars=[];checks=[]
    for spec in p['test']:
        folder=folders[spec['name']];actual=json.loads((folder/'parameters.json').read_text())
        for key,value in spec.items():assert actual[key]==value,(spec['name'],key)
        assert (folder/'metrics.json').exists() and not list(folder.glob('*.partial.hdf5'))
        row=completed_row(folder,CHECKPOINT_SHA)
        decisions=json.loads((folder/'policy-decisions.json').read_text());samples=json.loads((folder/'samples.json').read_text())
        tipping=[s for s in samples if abs(s['bottle_tilt_deg'])>20.]
        contact_fraction=sum(s['both_fingers'] for s in tipping)/max(1,len(tipping))
        assert abs(contact_fraction-row['sustained_grasp_fraction'])<1e-12
        assert row['grasp_verified']==any(s['grasp_verified'] for s in samples)
        assert row['final_outside_ml']==samples[-1]['spilled_ml']+samples[-1]['airborne_ml']
        assert row['ever_below_worktop_ml']==max(s.get('ever_table_ml',0.) for s in samples)
        assert row['zero_spill']==(row['final_outside_ml']==0 and row['ever_below_worktop_ml']==0)
        assert row['cup_surface_height_fraction']==samples[-1]['cup_fill_fraction']
        assert row['source_remaining_ml']==samples[-1]['bottle_ml']
        ordered_modes=('Returning upright','Setting bottle down','Releasing bottle','Retreating','Holding completed pose')
        times=[next((d['t'] for d in decisions if d['stop_latched'] and d['mode']==mode),None) for mode in ordered_modes]
        ordered=all(t is not None for t in times) and all(a<b for a,b in zip(times,times[1:]))
        assert row['completed_return_release_sequence']==bool(ordered and row['policy_completed'])
        goal=row['cup_surface_height_fraction']>=.9 if row['expected_stop']=='cup_near_full' else row['source_remaining_ml']<=max(.25,.01*spec['initial_ml'])
        strict=bool(row['grasp_verified'] and row['visual_grasp_confirmed'] and row['sustained_grasp_fraction']>=.8 and
            row['delivered_to_cup_ml']>=1. and row['stop_reason']==row['expected_stop'] and goal and
            ordered and row['policy_completed'] and row['zero_spill'] and not row['gates'] and
            abs(samples[-1]['bottle_tilt_deg'])<8 and abs(samples[-1]['bottle_base_height_m'])<=.005)
        row.update(strict_pass=strict,layout_label=spec['layout_label'],layout_index=spec['layout_index'],
            bottle_world_xy=[spec['cup_world'][0]-spec['source_xy'][1],spec['cup_world'][1]+spec['source_xy'][0]])
        row['localization_scores']=dict(
            source_threshold=.7,cup_threshold=.6314167022705078,
            maximum_reported_source_score=max(d['source_localization_score'] for d in decisions if d.get('source_localization_score') is not None),
            maximum_reported_cup_score=max(d['cup_localization_score'] for d in decisions if d.get('cup_localization_score') is not None),
            final_consensus=decisions[-1].get('localization_consensus'))
        row['failure_reasons']=[] if strict else failure_reasons(row)
        early=[s for s in samples if s['t']<=.30 and any('gripper' in c for c in s['contacts'])]
        row['initial_bottle_hand_contact']=bool(early)
        row['initial_bottle_hand_contact_time_s']=early[0]['t'] if early else None
        row['initial_geometry_overlap']=geometry.get(row['name'],{}).get('initial_envelope_intersections_including_fingers',[])
        row['setup_flag']=bool(early or row['initial_geometry_overlap'])
        row['setup_evidence_note']='Early native bottle/finger contact' if early else 'Post-hoc convex-envelope geometry flag; cup contact was not instrumented' if row['initial_geometry_overlap'] else 'No initial full-hand geometry flag or early recorded bottle/hand contact'
        if row['setup_flag']:row['failure_reasons'].insert(0,row['setup_evidence_note'])
        assert strict or row['failure_reasons']
        quantum=spec['spacing']**3*1e6
        for s in samples:
            inventory=sum(s[k] for k in ('bottle_ml','cup_ml','airborne_ml','spilled_ml'))/quantum
            assert abs(inventory-s['count'])<1e-5
        ep=folder/'episode.hdf5';check=validate(ep);check.pop('metrics')
        with h5py.File(ep,'r') as f:
            assert f.attrs['evidence_kind']=='learned_visual_policy'
            assert json.loads(f.attrs['policy_camera_names'])==['task','wrist']
            d=f['data/demo_0'];n=check['transitions'];max_skew=0.
            for group,timekey in [('obs','time'),('next_obs','next_time')]:
                ds=d[group+'/overview_image'];assert ds.shape==(n,)
                skew=float(np.max(abs(d[group+'/overview_time'][:]-d[timekey][:])));max_skew=max(max_skew,skew);assert skew<1e-6
                assert (np.diff(d[group+'/overview_time'][:])>0).all()
                for i in range(n):
                    frame=decode(ds[i],'jpeg95');assert frame.shape==(384,384,3) and frame.std()>5.
            for i in range(1,n):assert np.array_equal(d['obs/overview_image'][i],d['next_obs/overview_image'][i-1])
            assert np.allclose(d['obs/overview_world_T_camera'][:,3,:],[0,0,0,1])
            check['overview']=dict(frames=n,decoded_images=n*2,resolution=[384,384],max_camera_state_skew_s=max_skew)
        check['particle_inventory_samples']=len(samples);checks.append(check)
        files.append(item(ep));sidecars.extend(item(q) for q in sorted(folder.glob('*.json')))
        rows.append(row)
        print(json.dumps(dict(name=row['name'],pass_=strict,reason=row['stop_reason'],frames=check['transitions'])),flush=True)
    counts=dict(planned=len(p['test']),complete=len(rows),strict_passes=sum(r['strict_pass'] for r in rows),
        verified_grasps=sum(r['grasp_verified'] for r in rows),pours=sum(r['delivered_to_cup_ml']>=1 for r in rows),
        zero_measured_loss=sum(r['zero_spill'] for r in rows),localization_aborts=sum(r['stop_reason']=='localization_unconfirmed' for r in rows),
        setup_flags=sum(r['setup_flag'] for r in rows),unflagged_starting_layouts=sum(not r['setup_flag'] for r in rows))
    report=dict(status='complete',counts=counts,cases=rows,selection=selection,
        scope=p['scope'],primary_definition=selection['primary_definition'],
        note='Every planned once-only autonomous test included. Initial setup flags are explicit; no-pour zero-loss aborts are failures. Nominal mL derives from particle counts, not calibrated physical volume. Observations20Hz, physics400Hz.')
    (ROOT/'final-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'recording-integrity.json').write_text(json.dumps(dict(files=files,sidecar_files=sidecars,checks=checks,
        partial_files=[],native_checks_passed=True,recordings=len(files),transitions=sum(c['transitions'] for c in checks)),indent=2)+'\n')
    with (ROOT/'final-benchmark.csv').open('w',newline='') as f:
        keys=['name','layout_label','strict_pass','expected_stop','stop_reason','grasp_verified','delivered_to_cup_ml','cup_surface_height_fraction','source_remaining_ml','final_outside_ml','failure_reasons']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in keys} for r in rows)
    print(json.dumps(counts),flush=True)

if __name__=='__main__': main()
