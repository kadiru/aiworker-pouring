"""Five PBD fluids: physical readback, smooth surfaces, equal-volume comparison.

Separate systems allow different materials but do not interact with each other.
Comparison spills therefore run one at a time at the same position.
"""
import numpy as np
from pour_geometry import LIQUIDS, SOURCE_BOTTOM, CUP_INTERIORS, TABLE_BOUNDS, NOMINAL_VOLUME_M3


def seed_points(spec):
    s = spec['spacing']
    count = round(NOMINAL_VOLUME_M3 / s**3)
    limit = int(.021 / s)
    cross = [(ix*s, iy*s) for ix in range(-limit,limit+1) for iy in range(-limit,limit+1)
             if (ix*s)**2+(iy*s)**2 <= .021**2]
    cross.sort(key=lambda p:p[0]**2+p[1]**2)
    return np.array([np.array(spec['position']) + [*cross[i%len(cross)], SOURCE_BOTTOM+.006+(i//len(cross))*s]
                     for i in range(count)], dtype=np.float32)


def metadata():
    offset = 0
    specs = []
    for spec in LIQUIDS:
        n = len(seed_points(spec))
        specs.append(dict(spec, particle_count=n, particle_radius_m=spec['spacing']/2,
                          nominal_volume_ml=n*spec['spacing']**3*1e6,
                          recording_slice=[offset,offset+n]))
        offset += n
    return specs


def surface_profile(env, legacy=False):
    from pxr import PhysxSchema
    for f in env._pour_fluids:
        s = f['spec']['spacing']
        api = PhysxSchema.PhysxParticleIsosurfaceAPI(env.sim.stage.GetPrimAtPath(f['system']))
        api.CreateSurfaceDistanceAttr(s*(.75 if legacy else .9))
        api.CreateGridFilteringPassesAttr('GSRS' if legacy else 'GS')
        api.CreateGridSmoothingRadiusAttr(s*(1.5 if legacy else .75))
        api.CreateNumMeshSmoothingPassesAttr(4 if legacy else 2)
        api.CreateMaxSubgridsAttr(4096 if f['spec']['id']==1 else 2048)
        api.CreateMaxVerticesAttr(131072)
        api.CreateMaxTrianglesAttr(262144)
    env._pour_surface_profile = 'legacy' if legacy else 'thin_spill'
    apply_water_look(env)
    install_surface_cadence(env)


def install_surface_cadence(env):
    """Keep 200 Hz collision solving; rebuild surface once per rendered action."""
    import omni.physx
    context = env.sim.get_physics_context()
    direct_step = type(context)._step.__get__(context, type(context))
    interface = omni.physx.get_physx_interface()
    context._pour_substep = 0
    def step(current_time, update_fabric=False):
        context._pour_substep += 1
        if context._pour_substep % env.cfg.sim.render_interval == 0:
            interface.update_simulation(context.get_physics_dt(), current_time)
            interface.update_transformations(False, True, True, False)
        else:
            direct_step(current_time, update_fabric)
    context._step = step


def apply_water_look(env):
    """Transparent inspection material; physics is independent of this display."""
    from pxr import UsdShade, UsdGeom, UsdLux, Sdf, Gf, Vt
    stage = env.sim.stage
    water = UsdShade.Material.Get(stage,'/World/PourWaterSurface')
    # This headless XR renderer made the MDL transmission either dark or nearly
    # invisible. Use an explicit transparent preview surface for inspection.
    water.GetPrim().RemoveProperty('outputs:mdl:surface')
    shader = UsdShade.Shader.Define(stage,'/World/PourWaterSurface/PreviewShader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12,.42,.62))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.06)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(.14)
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.333)
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.08,.1,.12))
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    water.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    if not stage.GetPrimAtPath('/World/WaterInspectionLight'):
        light = UsdLux.RectLight.Define(stage,'/World/WaterInspectionLight')
        light.CreateWidthAttr(.35); light.CreateHeightAttr(.12)
        light.CreateIntensityAttr(2500.)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(.8,.3,1.65))
    if not stage.GetPrimAtPath('/World/SpillContrastGrid'):
        mesh = UsdGeom.Mesh.Define(stage,'/World/SpillContrastGrid')
        vertices=[]; indices=[]; colors=[]
        for ix in range(8):
            for iy in range(8):
                x=.78+ix*.04; y=-.16+iy*.04
                indices.extend(range(len(vertices),len(vertices)+4))
                vertices.extend([(x,y,1.0001),(x+.04,y,1.0001),(x+.04,y+.04,1.0001),(x,y+.04,1.0001)])
                colors.append(Gf.Vec3f(.18,.22,.26) if (ix+iy)%2 else Gf.Vec3f(.38,.42,.46))
        mesh.CreatePointsAttr(Vt.Vec3fArray(vertices)); mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateFaceVertexCountsAttr([4]*64)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(colors))


def spawn_fluid(env, env_ids=None):
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Sdf, Gf, UsdGeom, Vt, PhysxSchema, UsdShade
    stage = env.sim.stage
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(env.cfg.sim.physics_prim_path))
    scene_api.CreateEnableGPUDynamicsAttr(True)
    scene_api.CreateBroadphaseTypeAttr('GPU')
    water = UsdShade.Material.Define(stage, '/World/PourWaterSurface')
    shader = UsdShade.Shader.Define(stage, '/World/PourWaterSurface/Shader')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath('/isaac-sim/kit/mdl/core/Base/OmniSurfacePresets.mdl'), 'mdl')
    shader.SetSourceAssetSubIdentifier('OmniSurface_ClearWater', 'mdl')
    shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
    water.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')
    env._pour_fluids = []
    for spec in LIQUIDS:
        s = spec['spacing']
        system = Sdf.Path(f"/World/PourParticleSystem{spec['id']}")
        material = Sdf.Path(f"/World/PourPhysicsMaterial{spec['id']}")
        path = f"/World/PourFluid{spec['id']}"
        contact = s/(2*.99*.6)
        particleUtils.add_physx_particle_system(stage, system, contact_offset=contact, rest_offset=contact*.99,
            particle_contact_offset=contact, solid_rest_offset=contact*.99, fluid_rest_offset=s/2,
            solver_position_iterations=16, simulation_owner=env.cfg.sim.physics_prim_path, max_neighborhood=96,
            enable_ccd=True, max_depenetration_velocity=1., max_velocity=3.,
            global_self_collision_enabled=True, non_particle_collision_enabled=True)
        particleUtils.add_pbd_particle_material(stage, material, cohesion=0., viscosity=spec['viscosity'],
            surface_tension=0., friction=.08, damping=0., vorticity_confinement=0.)
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system), material)
        # Preserve thin sheets: larger sparse grid budget, no reduce pass.
        # Particle smoothing stays disabled so positions are physical readback.
        particleUtils.add_physx_particle_isosurface(stage, system, enabled=True,
            max_vertices=131072, max_triangles=262144, max_subgrids=4096 if spec['id']==1 else 2048,
            grid_spacing=s*.75, surface_distance=s*.9, grid_filtering_passes='GS',
            grid_smoothing_radius=s*.75, num_mesh_smoothing_passes=2, num_mesh_normal_smoothing_passes=4)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(system)).Bind(water)
        points = seed_points(spec)
        prim = particleUtils.add_physx_particleset_pointinstancer(
            stage, Sdf.Path(path), [Gf.Vec3f(*map(float,p)) for p in points], [Gf.Vec3f(0)]*len(points),
            system, self_collision=True, fluid=True, particle_group=spec['id'],
            particle_mass=1000*s**3, density=0., num_prototypes=1)
        instancer = UsdGeom.PointInstancer(prim)
        prototype = UsdGeom.Sphere.Define(stage, path+'/particlePrototype0')
        prototype.CreateRadiusAttr(s/2)
        prototype.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*spec['rgb'])]))
        prototype.CreateVisibilityAttr('invisible')
        instancer.GetProtoIndicesAttr().Set([0]*len(points))
        env._pour_fluids.append(dict(spec=spec, initial=points, instancer=instancer, system=str(system), points_path=path))
    env._pour_surface_profile = 'thin_spill'
    env._pour_metadata = metadata()
    apply_water_look(env)


def set_positions(fluid, points):
    from pxr import Gf, Vt
    fluid['instancer'].GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float,p)) for p in points]))
    fluid['instancer'].GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0)]*len(points)))
    fluid['previous'] = np.asarray(points).copy()
    fluid['ever_past_edge'] = np.zeros(len(points), dtype=bool)
    fluid['suspected_crossings'] = np.zeros(len(points), dtype=bool)


def reset_fluid(env, env_ids=None):
    for f in getattr(env, '_pour_fluids', []): set_positions(f, f['initial'])


def spill(env, index):
    """Synthetic standardized drop, not a robot pour; no particles added/deleted."""
    f = env._pour_fluids[index]
    p = fluid_positions(f)
    p[:,:2] += np.array([.94,0])-np.array(f['spec']['position'][:2])
    p[:,2] += 1.08-p[:,2].min()
    set_positions(f, p)


def fluid_positions(f):
    p = f['instancer'].GetPositionsAttr().Get()
    if p is None or len(p) != len(f['initial']): raise RuntimeError('Physical particle count changed')
    p = np.asarray(p, dtype=np.float32).copy()
    if not np.isfinite(p).all(): raise RuntimeError('Non-finite PBD state')
    return p


def positions(env):
    return np.concatenate([fluid_positions(f) for f in env._pour_fluids])


def footprint(p):
    xmin,xmax,ymin,ymax = TABLE_BOUNDS
    return (p[:,0]>=xmin)&(p[:,0]<=xmax)&(p[:,1]>=ymin)&(p[:,1]<=ymax)


def diagnostics(env):
    from scipy.spatial.transform import Rotation
    entries = []
    aggregate = {name:0 for name,*_ in CUP_INTERIORS}
    for f,spec in zip(env._pour_fluids, env._pour_metadata):
        p = fluid_positions(f)
        incup = np.zeros(len(p), dtype=bool)
        cups = {}
        for name,radius,bottom,top in CUP_INTERIORS:
            pose = env.scene[name].data.root_state_w[0,:7].cpu().numpy()
            local = Rotation.from_quat(pose[[4,5,6,3]]).inv().apply(p-pose[:3])
            mask = (np.square(local[:,:2]).sum(axis=1)<=radius**2)&(local[:,2]>=bottom-.004)&(local[:,2]<=top+.004)
            cups[name] = int(mask.sum())
            aggregate[name] += cups[name]
            incup |= mask
        over = footprint(p)
        previous = f.get('previous',p)
        ever_edge = f.setdefault('ever_past_edge',np.zeros(len(p),dtype=bool))
        crossings = f.setdefault('suspected_crossings',np.zeros(len(p),dtype=bool))
        # A later ground bounce can move an edge spill under the table. Count
        # downward crossings separately, away from the slab's edge.
        interior = (p[:,0]>.345)&(p[:,0]<1.155)&(np.abs(p[:,1])<.505)
        crossings |= (previous[:,2]>=.99)&(p[:,2]<.99)&interior&~ever_edge
        ever_edge |= ~over
        f['previous'] = p.copy()
        tabletop = over & ~incup & (p[:,2]>=.995)&(p[:,2]<=1.025)
        entries.append(dict(spec, cups=cups, total=len(p), outside_cups=int((~incup).sum()),
            on_table=int(tabletop.sum()), under_table_footprint=int((over&(p[:,2]<.99)).sum()),
            over_edge=int((~over).sum()), below_table=int((p[:,2]<.99).sum()),
            ever_past_edge=int(ever_edge.sum()), suspected_table_crossings=int(crossings.sum()),
            z_min_m=float(p[:,2].min()), z_max_m=float(p[:,2].max()),
            tabletop_extent_m=np.ptp(p[tabletop,:2],axis=0).tolist() if tabletop.any() else [0,0],
            tabletop_center_m=p[tabletop].mean(axis=0).tolist() if tabletop.any() else None))
    aggregate.update(total=sum(f['total'] for f in entries), outside_both=sum(f['outside_cups'] for f in entries),
                     on_table=sum(f['on_table'] for f in entries), below_table=sum(f['below_table'] for f in entries),
                     under_table_footprint=sum(f['under_table_footprint'] for f in entries), liquids=entries)
    return aggregate


def render_diagnostics(env):
    from pxr import UsdGeom, PhysxSchema, UsdShade
    from scipy.spatial import cKDTree
    import carb
    entries = []
    for f in env._pour_fluids:
        api = PhysxSchema.PhysxParticleIsosurfaceAPI(env.sim.stage.GetPrimAtPath(f['system']))
        mesh = UsdGeom.Mesh.Get(env.sim.stage, f['system']+'/Isosurface')
        points = mesh.GetPointsAttr().Get() if mesh else None
        vertices = np.asarray(points) if points is not None and len(points) else np.empty((0,3))
        p = fluid_positions(f)
        table = footprint(p)&(p[:,2]>=.995)&(p[:,2]<=1.006)
        covered = int((cKDTree(vertices).query(p[table])[0] < 3*f['spec']['spacing']).sum()) if len(vertices) and table.any() else 0
        faces = mesh.GetFaceVertexCountsAttr().Get() if mesh else None
        entries.append(dict(id=f['spec']['id'], surface_distance_m=api.GetSurfaceDistanceAttr().Get(),
            grid_filter=api.GetGridFilteringPassesAttr().Get(), max_subgrids=api.GetMaxSubgridsAttr().Get(),
            mesh_vertices=len(vertices), mesh_faces=len(faces) if faces is not None else 0,
            mesh_visible=bool(mesh and mesh.ComputeVisibility()!='invisible'),
            particle_spheres_visible=UsdGeom.Imageable(env.sim.stage.GetPrimAtPath(f['points_path']+'/particlePrototype0')).ComputeVisibility()!='invisible',
            mesh_above_table_vertices=int(((vertices[:,2]>1.0005)&(vertices[:,2]<1.025)).sum()),
            tabletop_particles=int(table.sum()), tabletop_particles_near_mesh=covered,
            mesh_extent_m=np.ptp(vertices,axis=0).tolist() if len(vertices) else None))
    shader = UsdShade.Shader.Get(env.sim.stage,'/World/PourWaterSurface/PreviewShader')
    return dict(method='PhysX isosurface; transparent preview material', profile=env._pour_surface_profile,
                measurement_positions='raw physical positions; smoothing disabled', liquids=entries,
                translucency_enabled=carb.settings.get_settings().get('/rtx/translucency/enabled'),
                reflections_enabled=carb.settings.get_settings().get('/rtx/reflections/enabled'),
                physics_hz=1/env.cfg.sim.dt, surface_hz=1/(env.cfg.sim.dt*env.cfg.sim.render_interval),
                opacity=shader.GetInput('opacity').Get(), ior=shader.GetInput('ior').Get())
