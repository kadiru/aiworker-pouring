"""Heuristic fluid diagnostics from unsmoothed physical points, with explicit gates.

No metric here is a calibrated probability of being water. Neighbor and wall
counts are geometric proxies, not native solver contact impulses.
"""
import numpy as np
from scipy.spatial import cKDTree

HALF_X = .05
HALF_Y = .04
WALL_HEIGHT = .075


def sample(points, velocities, spec, t, expected_count, emitted_ml, initial_speed, mesh=None):
    p = np.asarray(points).reshape(-1,3)
    v = np.asarray(velocities).reshape(-1,3)
    s = spec['spacing']
    finite = bool(np.isfinite(p).all() and np.isfinite(v).all())
    if not finite or len(p) != len(v):
        return dict(t=t, count=len(p), expected_count=expected_count, finite=False)
    inside = (abs(p[:,0]) < HALF_X) & (abs(p[:,1]) < HALF_Y)
    pool = inside & (p[:,2] > -.001) & (p[:,2] < WALL_HEIGHT)
    stream = (p[:,2] > WALL_HEIGHT+.015) & (p[:,2] < spec['height']-.01)
    # Particle centers must stay above the receiver's floor top (z=0).
    # A 1 mm tolerance excludes numerical noise; the slab is 12 mm thick.
    floor_interior = (abs(p[:,0]) < HALF_X-2*s) & (abs(p[:,1]) < HALF_Y-2*s)
    below = floor_interior & (p[:,2] < -.001)
    escaped = (~inside) & (p[:,2] < WALL_HEIGHT)
    speed = np.linalg.norm(v,axis=1)
    near_apparatus=(abs(p[:,0])<.18)&(abs(p[:,1])<.15)&(p[:,2]>-.05)
    pp = p[pool]
    pv = v[pool]
    occupancy = 0.; surface_std = 0.; mound = 0.
    if len(pp):
        xbin = np.clip(((pp[:,0]+HALF_X)/(2*HALF_X)*8).astype(int),0,7)
        ybin = np.clip(((pp[:,1]+HALF_Y)/(2*HALF_Y)*6).astype(int),0,5)
        tops = [np.quantile(pp[(xbin==ix)&(ybin==iy),2], .9)
                for ix in range(8) for iy in range(6)
                if np.count_nonzero((xbin==ix)&(ybin==iy)) >= 2]
        occupancy = len(tops)/48
        surface_std = float(np.std(tops)) if tops else 0.
        mound = float(np.quantile(pp[:,2],.95)-np.median(pp[:,2]))
    jet_continuity = 0.; radius_cv = 0.; ballistic_error = 0.
    if stream.any():
        jet = p[stream]; jv = v[stream]
        edges = np.linspace(WALL_HEIGHT+.015,spec['height']-.01,9)
        slices = [(jet[:,2]>=a)&(jet[:,2]<b) for a,b in zip(edges[:-1],edges[1:])]
        jet_continuity = sum(np.count_nonzero(m)>=2 for m in slices)/8
        radii = [float(np.sqrt(np.mean(np.sum((jet[m,:2]-np.mean(jet[m,:2],axis=0))**2,axis=1))))
                 for m in slices if m.any()]
        radius_cv = float(np.std(radii)/(np.mean(radii)+1e-9)) if radii else 1.
        expected_speed = np.sqrt(initial_speed**2+2*9.81*(spec['height']-jet[:,2]))
        ballistic_error = float(np.median(abs(-jv[:,2]-expected_speed)/np.maximum(expected_speed,.1)))
    isolated = 0.; neighbors = 0.; near_wall = 0
    if len(p)>1:
        tree = cKDTree(p)
        counts = tree.query_ball_point(p,2*s,return_length=True)-1
        isolated = float(np.mean(counts==0))
        neighbors = float(np.mean(counts))
    if pool.any():
        near_wall = int(np.count_nonzero(pool & ((HALF_X-abs(p[:,0])<2*s) | (HALF_Y-abs(p[:,1])<2*s) | (p[:,2]<2*s))))
    coverage = None
    pool_mesh_coverage = None
    if mesh is not None and len(mesh) and len(p):
        distances = cKDTree(np.asarray(mesh)).query(p,workers=1)[0]
        coverage = float(np.mean(distances<3*s))
        pool_mesh_coverage = float(np.mean(distances[pool]<3*s)) if pool.any() else None
    return dict(t=round(t,4), count=len(p), expected_count=expected_count, finite=finite,
                emitted_ml=emitted_ml, pool_count=int(pool.sum()), leak_count=int(below.sum()),
                escaped_count=int(escaped.sum()), jet_count=int(stream.sum()),
                pool_volume_ml=float(pool.sum()*s**3*1e6),
                retention=float(pool.sum()/max(len(p),1)),
                pool_coverage=occupancy, surface_std_mm=surface_std*1000, mound_mm=mound*1000,
                pool_com_x=float(np.mean(pp[:,0])) if len(pp) else 0.,
                pool_rms_speed=float(np.sqrt(np.mean(np.sum(pv*pv,axis=1)))) if len(pv) else 0.,
                speed_p99=float(np.quantile(speed[near_apparatus],.99)) if near_apparatus.any() else 0.,
                all_particle_speed_p99=float(np.quantile(speed,.99)) if len(speed) else 0.,
                jet_continuity=jet_continuity, jet_radius_cv=radius_cv,
                ballistic_error=ballistic_error, isolated_fraction=isolated,
                mean_neighbors=neighbors, wall_proximity_count=near_wall,
                mesh_particle_coverage=coverage,pool_mesh_coverage=pool_mesh_coverage)


def slosh_response(samples,spec):
    rows=[r for r in samples if r.get('finite') and r['t']>spec['slosh_time']+.15]
    t=np.array([r['t']-spec['slosh_time'] for r in rows])
    y=np.array([r['pool_com_x'] for r in rows])
    if len(t)<20: return dict(valid=False,reason='insufficient_samples')
    # Fit offset + exp(-gamma*t)*(a*sin(omega*t)+b*cos(omega*t)).
    # A bounded grid avoids fragile nonlinear initialization on a short record.
    best=None
    for hz in np.linspace(.35,2.5,87):
        for decay in np.linspace(0.,4.,41):
            envelope=np.exp(-decay*t)
            design=np.column_stack((envelope*np.sin(2*np.pi*hz*t),envelope*np.cos(2*np.pi*hz*t),np.ones(len(t))))
            coeff=np.linalg.lstsq(design,y,rcond=None)[0]
            residual=float(np.mean((y-design@coeff)**2))
            if best is None or residual<best[0]: best=(residual,float(hz),float(decay),coeff)
    residual,hz,decay,coeff=best
    fit_r2=float(1-residual/max(float(np.var(y)),1e-12))
    amplitude=float(np.hypot(coeff[0],coeff[1]))
    # Ideal inviscid rectangular-basin first mode; an approximate sanity check.
    depth=rows[-1]['pool_volume_ml']*1e-6/(4*HALF_X*HALF_Y)
    k=np.pi/(2*HALF_X)
    ideal_hz=float(np.sqrt(9.81*k*np.tanh(k*depth))/(2*np.pi))
    response=dict(frequency_hz=hz,decay_per_s=decay,amplitude_mm=amplitude*1000,
                  fit_r2=fit_r2,ideal_inviscid_hz=ideal_hz,
                  frequency_ratio=hz/max(ideal_hz,1e-6),
                  valid=bool(fit_r2>.5 and amplitude>.0002 and .5<hz/max(ideal_hz,1e-6)<1.5))
    response['score']=float(100*max(0.,fit_r2)*np.exp(-abs(np.log(hz/max(ideal_hz,1e-6)))/.4)*np.exp(-max(0.,decay-.25)/1.2))
    return response


def summarize(samples, spec):
    good = [r for r in samples if r.get('finite')]
    gates = []
    if len(good)!=len(samples) or not good: gates.append('nonfinite')
    if any(r['count'] != r['expected_count'] for r in samples): gates.append('particle_count_changed')
    if not good: return dict(eligible=False,gates=gates,screening_score=0.)
    peak_leak = max(r['leak_count'] for r in good)
    final = good[-1]
    if peak_leak: gates.append('floor_leak')
    if final['emitted_ml'] < spec['flow_ml_s']*spec['flow_seconds']*.95: gates.append('under_emission')
    if max(r['speed_p99'] for r in good)>5.: gates.append('unstable_speed')
    if final['retention']<.9: gates.append('poor_retention')
    if spec['name'].startswith('control_'): gates.append('diagnostic_control')
    flowing = [r for r in good if .8<r['t']<spec['flow_seconds']-.1]
    tail = [r for r in good if r['t']>spec['flow_seconds']+spec['settle_seconds']-.5]
    early = [r for r in good if spec['flow_seconds']+.3<r['t']<spec['flow_seconds']+.8]
    def avg(rows,key): return float(np.mean([r[key] for r in rows])) if rows else 0.
    active_speed = avg(flowing,'pool_rms_speed')
    final_speed = avg(tail,'pool_rms_speed')
    continuity = avg(flowing,'jet_continuity')
    ballistic = avg(flowing,'ballistic_error')
    spread = avg(tail,'pool_coverage')
    roughness = avg(tail,'surface_std_mm')
    # Reward an active pool that spreads and settles. All weights are hypotheses.
    # Absolute final speed prevents a wildly energetic flow exploiting decay ratio.
    components = dict(retention=final['retention'], jet=continuity,
        free_fall=max(0.,1.-ballistic), spread=spread,
        level=float(np.exp(-roughness/3.)),
        settling=float(np.exp(-final_speed/.025)),
        flow_motion=min(1.,active_speed/.05))
    weights = dict(retention=.20, jet=.15, free_fall=.10, spread=.15, level=.15, settling=.15, flow_motion=.10)
    score = 100*sum(components[k]*w for k,w in weights.items())
    result=dict(eligible=not gates, gates=gates, screening_score=round(score,3), components=components,
                count=final['count'], emitted_ml=final['emitted_ml'], final_retention=final['retention'],
                peak_leak_count=peak_leak, jet_continuity=continuity, ballistic_error=ballistic,
                final_coverage=spread, final_surface_std_mm=roughness,
                flowing_rms_m_s=active_speed, final_rms_m_s=final_speed,
                decay_ratio=final_speed/max(avg(early,'pool_rms_speed'),1e-6),
                final_mesh_coverage=final['mesh_particle_coverage'],
                final_pool_mesh_coverage=final.get('pool_mesh_coverage'),
                metrics_version=2,
                caveat='Heuristic physics screening, not calibrated waterness; wall/neighbor counts are proxies.')
    if spec.get('slosh_speed',0)>0:
        result['slosh']=slosh_response(good,spec)
        result['water_screening_score']=round(.5*score+.5*result['slosh'].get('score',0.),3)
        if not result['slosh']['valid']:
            result['gates'].append('weak_slosh_response');result['eligible']=False
    return result
