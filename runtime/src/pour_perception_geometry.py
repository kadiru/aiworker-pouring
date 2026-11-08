"""Static calibration geometry and robust training labels, never live policy truth."""
import numpy as np


def calibration_seed_points(spec, seed):
    from pour_vessel_geometry import cup_profile
    from bottle_pour import SIDES
    assert spec.get('calibration_only') and not spec.get('policy_evaluation')
    fraction=float(spec['calibration_initial_height'])
    if not .15 <= fraction <= 1.:
        raise ValueError('Calibration seed cut must be in [.15, 1]')
    profile=cup_profile(spec);spacing=spec['spacing'];p=np.asarray(seed,dtype=np.float32)
    assert p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all()
    z=p[:,2];angle=(np.arctan2(p[:,1],p[:,0])%(2*np.pi/SIDES))-np.pi/SIDES
    radius=np.interp(z,[v[0] for v in profile],[v[1] for v in profile])*np.cos(np.pi/SIDES)/np.cos(angle)
    top=profile[0][0]+fraction*(profile[-1][0]-profile[0][0])-spacing/2
    keep=(z>=profile[0][0])&(z<=top)&(np.linalg.norm(p[:,:2],axis=1)<radius-.5*spacing)
    if keep.sum()<40:raise ValueError('Insufficient seed support for a measured surface')
    return p[keep].copy()


def upper_surface(points,profile,spacing):
    """Robust central free-surface diagnostic; never changes legacy task scores.

    Spatial cells each contribute one highest particle, so changing the fraction
    of particles in a top layer cannot switch a volume quantile to the next layer.
    Median aggregation rejects isolated high droplets in a minority of cells.
    """
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or spacing<=0:
        raise ValueError('Invalid surface samples')
    pitch=2.5*spacing;radius=.75*min(v[1] for v in profile)
    index=np.floor(p[:,:2]/pitch).astype(int);tops=[]
    for key in np.unique(index,axis=0):
        center=(key+.5)*pitch
        if np.linalg.norm(center)+pitch/np.sqrt(2)>radius:continue
        values=p[np.all(index==key,axis=1),2]
        if len(values)>=3:tops.append(float(values.max()+spacing/2))
    if len(tops)<8:return dict(valid=False,cells=len(tops),height_fraction=None)
    bottom,rim=profile[0][0],profile[-1][0];value=(np.median(tops)-bottom)/(rim-bottom)
    return dict(valid=True,cells=len(tops),height_fraction=float(np.clip(value,0,1)),
        height_m=float(np.median(tops)),cell_top_iqr_m=float(np.quantile(tops,.75)-np.quantile(tops,.25)),
        pitch_m=pitch,central_radius_m=radius)


def settled_label(rows, seconds=1., field='cup_fill_fraction'):
    """Median from a stationary inventory plateau; reject drifting or lost fluid."""
    if not rows or not np.isfinite(seconds) or seconds<.5:
        raise ValueError('Need observations and at least half a second')
    fields=('t',field,'cup_ml','ever_table_ml','spilled_ml','airborne_ml')
    if not np.isfinite([[r[k] for k in fields] for r in rows]).all():
        raise ValueError('Nonfinite calibration measurement')
    if np.any(np.diff([r['t'] for r in rows])<=0):
        raise ValueError('Calibration timestamps must increase')
    last=rows[-1];tail=[r for r in rows if r['t']>=last['t']-seconds-1e-7]
    levels=np.asarray([r[field] for r in tail]);inventory=[r['cup_ml'] for r in tail]
    enough=len(tail)>=int(seconds/.05) and tail[-1]['t']-tail[0]['t']>=seconds-.051
    q1,q3=np.quantile(levels,[.25,.75]);half=len(tail)//2
    drift=float(abs(np.median(levels[:half])-np.median(levels[half:]))) if half else float('inf')
    stable=bool(enough and np.ptp(inventory)<=.005833 and q3-q1<=.01 and drift<=.005
                and max(r['ever_table_ml'] for r in rows)==0 and last['spilled_ml']+last['airborne_ml']==0)
    return dict(accepted=stable,target=float(np.median(levels)),field=field,window_s=seconds,samples=len(tail),
        height_iqr=float(q3-q1),half_window_drift=drift,height_min=float(levels.min()),height_max=float(levels.max()),
        final_sample=last[field],inventory_range_nominal_ml=float(np.ptp(inventory)),
        scope='Training-only stationary-window label; original instantaneous task scores remain unchanged.')
