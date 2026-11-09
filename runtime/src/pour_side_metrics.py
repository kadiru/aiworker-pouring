"""Independent recorded-state side-return checks; never imported by actor."""
import numpy as np

def side_metrics(samples,decisions,spec):
    plan=next((d.get('side_return_plan') for d in decisions if d.get('side_return_plan')),None)
    r=dict(planned=plan is not None,passed=False)
    if plan is None:return dict(r,reason='no_side_return_plan')
    cup=np.asarray(spec['cup_world'][:2]);target=np.asarray(plan['placement_xy'])
    final=np.asarray(samples[-1]['bottle_pose_xyz'][:2]);moving=rotating=0;vertical_over=0;start=None;upright=None
    for i,(s,d) in enumerate(zip(samples,decisions)):
        if d['mode']!='Returning upright' or not d['stop_latched']:continue
        if start is None:start=s
        if abs(s['bottle_tilt_deg'])<8 and upright is None:upright=s
        over=np.linalg.norm(np.asarray(s['bottle_pose_xyz'][:2])-cup)<.06
        vertical_over+=int(over and abs(s['bottle_tilt_deg'])<8)
        if i:
            prev=samples[i-1];dt=s['t']-prev['t']
            angular=abs(s['bottle_tilt_deg']-prev['bottle_tilt_deg'])/dt
            lateral=np.linalg.norm(np.asarray(s['bottle_pose_xyz'][:2])-prev['bottle_pose_xyz'][:2])/dt
            if angular>5:rotating+=1;moving+=int(lateral>.003)
    overlap=moving/max(rotating,1);sign=-1 if plan['side']=='right' else 1
    r.update(side=plan['side'],planned_placement_xy=target.tolist(),actual_placement_xy=final.tolist(),
        final_placement_error_m=float(np.linalg.norm(final-target)),
        final_lateral_offset_m=float(sign*(final[1]-cup[1])),
        rotation_samples=rotating,simultaneous_translation_rotation_fraction=overlap,
        upright_over_cup_samples=vertical_over,
        first_upright_cup_distance_m=None if upright is None else float(np.linalg.norm(np.asarray(upright['bottle_pose_xyz'][:2])-cup)),
        peak_surface_height_fraction=max(s['cup_fill_fraction'] for s in samples))
    r['passed']=bool(rotating>=5 and overlap>=.65 and vertical_over==0
        and r['final_placement_error_m']<=.015 and r['final_lateral_offset_m']>=.14
        and upright is not None and r['first_upright_cup_distance_m']>=.14)
    return r
