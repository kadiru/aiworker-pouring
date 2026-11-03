"""Scripted rigid-container hinge geometry and container-frame measurements.

The receiver is kinematic: this prescribes a repeatable lift/return, not a
gravity-driven drop or a measured table impact. No fluid velocity is injected.
"""
import math
import numpy as np

PIVOT = np.array([-.055, 0., -.012])
OBSTACLES = (
    dict(name='AmberBlock', center=[-.018, -.018, .014], size=[.012, .014, .028]),
    dict(name='BlueBlock', center=[.021, .017, .018], size=[.014, .014, .036]),
)


def hinge_trial(base, control=False):
    return dict(base, protocol=4, name='hinge_stationary_control' if control else 'hinge_obstacles',
                hinge=True, hinge_angle_deg=0. if control else 22., hinge_start=3.7,
                hinge_lift_seconds=.8, hinge_hold_seconds=.35,
                hinge_lower_seconds=.30, flow_seconds=3., settle_seconds=7.,
                slosh_speed=0., obstacles=True)


def motion(t, spec):
    """Positive angle lifts the opposite edge about the lower left Y-axis."""
    start=spec['hinge_start']; lift=spec['hinge_lift_seconds']
    hold=spec['hinge_hold_seconds']; lower=spec['hinge_lower_seconds']
    peak=math.radians(spec['hinge_angle_deg'])
    if t < spec['flow_seconds']: phase='Pouring'
    elif t < start: phase='Flow off; preparing lift'
    elif t < start+lift: phase='Lifting edge'
    elif t < start+lift+hold: phase='Holding tilt'
    elif t < start+lift+hold+lower: phase='Lowering to table'
    else: phase='Sloshing around obstacles'
    if t <= start: angle=rate=0.
    elif t < start+lift:
        u=(t-start)/lift
        angle=peak*(3*u*u-2*u*u*u); rate=peak*6*u*(1-u)/lift
    elif t < start+lift+hold: angle=peak; rate=0.
    elif t < start+lift+hold+lower:
        u=(t-start-lift-hold)/lower
        # Accelerating descent, then a prescribed hard stop at the table plane.
        angle=peak*(1-u*u); rate=-2*peak*u/lower
    else: angle=rate=0.
    if peak==0. and t>=spec['flow_seconds']: phase='Stationary control; observing pool'
    return angle,rate,phase


def touchdown(spec):
    return spec['hinge_start']+spec['hinge_lift_seconds']+spec['hinge_hold_seconds']+spec['hinge_lower_seconds']


def rotation(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,0.,-s],[0.,1.,0.],[s,0.,c]])


def to_container(points, velocities, angle, rate):
    """Return the level-reference coordinates and velocity relative to the body."""
    p=np.asarray(points); r=rotation(angle)
    local=(p-PIVOT)@r+PIVOT
    omega=np.array([0.,-rate,0.])
    relative=(np.asarray(velocities)-np.cross(omega,p-PIVOT))@r
    return local,relative


def obstacle_metrics(local_points, spacing):
    p=np.asarray(local_points)
    proximity=[]; penetration=[]
    for obstacle in OBSTACLES:
        q=abs(p-np.array(obstacle['center']))-np.array(obstacle['size'])/2
        distance=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
        proximity.append(int(np.count_nonzero(abs(distance)<2*spacing)))
        penetration.append(int(np.count_nonzero(distance < -.001)))
    return dict(obstacle_proximity_counts=proximity,obstacle_penetration_counts=penetration)


def floor_diagnostics(local_points, previous_local_points, spacing):
    """Check the finite slab and sampled downward crossings in the moving frame.

    An escaped droplet meters below the scene can project under the tilted floor
    without ever crossing it. Do not call that a floor leak.
    """
    p=np.asarray(local_points)
    inside=(abs(p[:,0])<.05-2*spacing)&(abs(p[:,1])<.04-2*spacing)
    intrusion=inside&(p[:,2]<-.001)&(p[:,2]>-.013)
    crossed=np.zeros(len(p),dtype=bool)
    if previous_local_points is not None:
        n=min(len(p),len(previous_local_points));old=np.asarray(previous_local_points)[:n];new=p[:n]
        ids=np.flatnonzero((old[:,2]>=-.001)&(new[:,2]<-.001))
        if len(ids):
            fraction=(-.001-old[ids,2])/(new[ids,2]-old[ids,2])
            hit=old[ids]+fraction[:,None]*(new[ids]-old[ids])
            crossed[ids]=(abs(hit[:,0])<.05-2*spacing)&(abs(hit[:,1])<.04-2*spacing)
    return dict(floor_intrusion_count=int(intrusion.sum()),floor_crossing_count=int(crossed.sum()),
                leak_count=int(np.count_nonzero(intrusion|crossed)),
                below_floor_plane_count=int(np.count_nonzero(inside&(p[:,2]<-.001))))


def response(samples,spec):
    stop=touchdown(spec)
    post=[r for r in samples if r['t']>=stop and r.get('finite')]
    before=[r for r in samples if spec['hinge_start']-.35<=r['t']<spec['hinge_start']]
    if not post: return dict(valid=False,reason='no_post_return_samples')
    x=np.array([r['pool_com_x'] for r in post])
    # COM reversals above 0.2 mm prominence exclude frame-to-frame noise.
    from scipy.signal import find_peaks
    peaks,_=find_peaks(x,prominence=.0002)
    troughs,_=find_peaks(-x,prominence=.0002)
    return dict(kind='kinematic edge lift and return; no fluid impulse',
                touchdown_seconds=stop,peak_angle_deg=max(r.get('hinge_angle_deg',0.) for r in samples),
                peak_edge_lift_mm=max(r.get('edge_lift_mm',0.) for r in samples),
                final_angle_deg=samples[-1].get('hinge_angle_deg'),
                pre_lift_rms_m_s=float(np.mean([r['pool_rms_speed'] for r in before])) if before else None,
                peak_post_return_rms_m_s=max(r['pool_rms_speed'] for r in post),
                post_return_com_range_mm=float(np.ptp(x)*1000),
                com_reversals_over_0p2mm=int(len(peaks)+len(troughs)),
                peak_obstacle_proximity=[max(r['obstacle_proximity_counts'][i] for r in post) for i in range(len(OBSTACLES))],
                peak_obstacle_penetration=[max(r['obstacle_penetration_counts'][i] for r in samples) for i in range(len(OBSTACLES))],
                peak_floor_intrusion=max(r.get('floor_intrusion_count',0) for r in samples),
                sampled_floor_crossings=sum(r.get('floor_crossing_count',0) for r in samples),
                caveat='Proximity and sampled COM reversals are geometric response proxies, not contact forces or a waterness score.')
