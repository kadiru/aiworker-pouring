"""Bounded PBD screening space; coefficient values are native PhysX units, not SI viscosity."""
BASE = dict(protocol=3,spacing=.0027, viscosity=.01, cohesion=0., surface_tension=0.,
            friction=.08, adhesion=0., damping=0., vorticity=0., iterations=16,
            neighbors=96, dt=.005, contact_ratio=1., flow_ml_s=25., height=.20,
            flow_seconds=3., settle_seconds=3., offset_x=.025, seed=42,
            surface_distance=.9, grid_spacing=.75, smoothing=.75,
            mesh_passes=2, opacity=.14, roughness=.06, collide=True,wall_friction=None)


def candidate(name, **changes):
    return dict(BASE, **changes, name=name)


def coarse_plan():
    # Anchors plus bounded joint exploration. Thick/leak controls are ineligible.
    result = [candidate('baseline'), candidate('control_thick', viscosity=100.),
              candidate('control_leak', collide=False),
              candidate('low_viscosity', viscosity=.001),
              candidate('very_low_viscosity', viscosity=.0000017),
              candidate('viscosity_0p1', viscosity=.1),
              candidate('viscosity_1', viscosity=1.),
              candidate('fine', spacing=.0018), candidate('coarse', spacing=.0036),
              candidate('low_friction', friction=0.),
              candidate('cohesion_small', cohesion=.001),
              candidate('tension_small', surface_tension=.001)]
    # Deterministic space-filling samples include interactions, without a huge grid.
    import random
    rng = random.Random(7301)
    axes = dict(viscosity=[.001,.003,.01,.03,.1,1.],
                cohesion=[0.,.0001,.001,.003,.01,.03],
                surface_tension=[0.,.000001,.00001,.0001,.001,.0074],
                friction=[0.,.01,.03,.06,.1,.2], adhesion=[0.,0.,.0001,.001,.003,.01],
                damping=[0.,0.,0.,.001,.01,.05], vorticity=[0.,0.,.001,.01,.03,.1])
    for values in axes.values():
        rng.shuffle(values)
    result.extend(candidate(f'joint_{i+1:02}', **{k:v[i] for k,v in axes.items()}) for i in range(6))
    return result


def refinement_plan(best):
    b = dict(best)
    b.pop('name')
    variants = [('refine_finer', dict(spacing=.0022)),
                ('refine_low_drag', dict(friction=.02, damping=0.)),
                ('refine_tension', dict(surface_tension=.0001)),
                ('refine_viscosity_down', dict(viscosity=max(1.7e-6,b['viscosity']/3))),
                ('refine_viscosity_up', dict(viscosity=min(3.,b['viscosity']*3)))]
    return [dict(b, **changes, name=name) for name,changes in variants]


def validation_plan(best):
    b = dict(best)
    b.pop('name')
    return [dict(b, name='validate_repeat', seed=43),
            dict(b, name='validate_density', mass_mode='density'),
            dict(b, name='validate_half_dt', dt=b['dt']/2, iterations=32, neighbors=128),
            dict(b, name='validate_high_flow', flow_ml_s=35., height=.25,
                 flow_seconds=min(3.,75000*b['spacing']**3*1e6/35.)),
            dict(b, name='validate_low_flow', flow_ml_s=15., height=.15)]


def slosh_plan(results):
    eligible=[r for r in results if r['spec'].get('protocol')==3 and r['metrics']['eligible'] and not r['spec']['name'].startswith(('slosh_','render_','validate_','replay_'))]
    ranked=sorted(eligible,key=lambda r:r['metrics']['screening_score'],reverse=True)
    choices=ranked[:3]
    # Include resolution and baseline anchors even when their preliminary score is lower.
    for name in ('baseline','fine','low_friction'):
        match=next((r for r in eligible if r['spec']['name']==name),None)
        if match and match not in choices: choices.append(match)
    plan=[dict(r['spec'],name='slosh_'+r['spec']['name'],slosh_speed=.08,slosh_time=3.5,settle_seconds=5.) for r in choices]
    plan.append(candidate('slosh_fine_400hz',spacing=.0018,dt=.0025,iterations=32,friction=.02,
                          slosh_speed=.08,slosh_time=3.5,settle_seconds=5.))
    plan.append(candidate('slosh_fine_800hz',spacing=.0018,dt=.00125,iterations=32,friction=.02,
                          slosh_speed=.08,slosh_time=3.5,settle_seconds=5.))
    plan.append(candidate('control_viscosity_10000',viscosity=10000.,slosh_speed=.08,slosh_time=3.5,settle_seconds=5.))
    return plan


def targeted_plan():
    b=dict(spacing=.0018,dt=.0025,iterations=16,friction=0.,viscosity=0.,wall_friction=0.,
           slosh_speed=.08,slosh_time=3.5,settle_seconds=5.)
    return [candidate('target_zero_drag',**b),
            candidate('target_dry_wall',**dict(b,wall_friction=None)),
            candidate('target_coarse_wet_wall',**dict(b,spacing=.0027)),
            candidate('target_neighbors_125',**dict(b,contact_ratio=1.25,neighbors=128)),
            candidate('target_neighbors_150',**dict(b,contact_ratio=1.5,neighbors=192)),
            candidate('target_cohesion_1',**dict(b,cohesion=1.)),
            candidate('target_cohesion_5',**dict(b,cohesion=5.,surface_tension=.02)),
            candidate('target_vorticity',**dict(b,vorticity=.1)),
            candidate('target_dense_fill',**dict(b,flow_ml_s=35.))]


def resolution_plan():
    b=dict(friction=0.,viscosity=0.,wall_friction=0.,slosh_speed=.08,slosh_time=3.5,settle_seconds=5.)
    return [candidate('resolution_1p8',**dict(b,spacing=.0018,dt=.0025,iterations=16)),
            candidate('resolution_1p2',**dict(b,spacing=.0012,dt=1/600,iterations=16)),
            candidate('resolution_1p0',**dict(b,spacing=.001,dt=.001,iterations=16))]


def render_plan(best):
    b=dict(best,name='render_default',slosh_speed=0.,settle_seconds=3.)
    return [b,dict(b,name='render_mdl',appearance='mdl'),
            dict(b,name='render_less_smoothing',smoothing=.4,mesh_passes=1),
            dict(b,name='render_finer_grid',grid_spacing=.5,surface_distance=.75,smoothing=.5),
            dict(b,name='render_clearer',opacity=.07,roughness=.025),
            dict(b,name='render_more_visible',opacity=.25,roughness=.04)]
