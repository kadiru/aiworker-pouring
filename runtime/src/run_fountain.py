"""One bounded, guarded GPU process for fountain screening and replay.

All simulation writes happen here on the main thread. Robot pours share this
process and allocation. Completed trials have parameters, diagnostics and frames.
"""
import json
import math
import os
import queue
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from fountain_plan import BASE, candidate, coarse_plan, refinement_plan, validation_plan, slosh_plan, render_plan, targeted_plan, resolution_plan
from fountain_hinge import PIVOT, OBSTACLES, hinge_trial, motion as hinge_motion, rotation as hinge_rotation, to_container, obstacle_metrics, floor_diagnostics, response as hinge_response
from quest_preview import Preview
from bottle_pour import Pour, bottle_spec

GPU = 'GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'
visible = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
if visible != [GPU]: raise RuntimeError(f'GPU isolation failed: {visible}')
RUN = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ROOT = Path('/workspace/outputs/fountain')
OUT = ROOT/RUN
OUT.mkdir(parents=True,exist_ok=True)
state = dict(ready=False, profile='fountain', phase='Starting Isaac Sim', paused=False,
             trial=None, results=[], queue=[], gpu_uuid=GPU, run=RUN,
             scoring='Heuristic physics screening; visual review is separate, not an automatic VLM score.')
allowed_commands={'recovery_collect','recovery_train','recovery_val','recovery_test','learning_train','learning_val','learning_test','learning_calibrate','learning_collect','robot_replay','vessels_review','props_a','props_b','props_c','props_frosted','props_clear','robot_full','robot_empty','bottle_full','bottle_empty','baseline','sweep','refine','validate','slosh','targeted','resolution','render','best','hinge','hinge_control','pause','stop','water','wine','close_view','wide_view','look_mdl','look_default','look_clearer','look_visible','audit_scene','hide_particles','show_particles'}
allowed_commands.add('recovery_probe')
allowed_commands.add('release_train')
allowed_commands.update(('stop_train','stop_val','stop_test'))
allowed_commands.update(('guard_train','guard_val','guard_regression','guard_development','guard_test'))
allowed_commands.update(('vote_development','vote_test'))
allowed_commands.add('robust_test')
allowed_commands.update(('settled_train','settled_development','settled_test'))
allowed_commands.update(('reliable_train','reliable_development','reliable_test'))
allowed_commands.update(('workspace_collect','workspace_train','workspace_development','workspace_test'))
allowed_commands.add('layout_variety_test')
allowed_commands.add('layout_variety_extra')
allowed_commands.update(('wide_pilot', 'wide_repair', 'wide_collect', 'wide_train', 'wide_development', 'wide_test'))
allowed_commands.update(('finish_train','finish_calibrate','finish_calibration_pilot','finish_development','finish_test'))
allowed_commands.update(('size_calibration','size_recovery_development'))
allowed_commands.add('size_seed_calibration')
allowed_commands.update(('perception_pilot','perception_collect','perception_development','perception_test'))
allowed_commands.add('perception_geometry')
allowed_commands.update(('perception_camera_pilot','perception_holdout'))
preview = Preview(state, directory='/workspace/fountain-viewer',allowed_commands=allowed_commands)

os.environ['OMNICLIENT_HUB_MODE']='disabled'
from isaacsim import SimulationApp
app = SimulationApp(dict(headless=True, width=1280, height=720,
                         window_width=1280, window_height=720, multi_gpu=False,
                         renderer='RayTracedLighting', anti_aliasing=3,limit_cpu_threads=8,
                         extra_args=['--reset-user','--/plugins/carb.tasking.plugin/threadCount=8',
                                     '--/plugins/omni.tbb.globalcontrol/maxThreadCount=8',
                                     '--/renderer/multiGpu/enabled=false',
                                     '--/log/file=/workspace/logs51/fountain-kit.log']))
import carb
import numpy as np
import omni.usd
import omni.physx
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, UsdLux, Vt, PhysxSchema
from omni.physx.scripts import particleUtils, physicsUtils
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.core.utils.viewports import set_camera_view
from fountain_metrics import sample, summarize, HALF_X, HALF_Y, WALL_HEIGHT

settings = carb.settings.get_settings()
for key,value in [('/physics/updateToUsd',True),('/physics/updateVelocitiesToUsd',True),
                  ('/physics/updateParticlesToUsd',True),('/physics/suppressReadback',False),
                  ('/rtx/translucency/enabled',True),('/rtx/reflections/enabled',True)]:
    settings.set(key,value)

sim = None
stage = None
water = None
current = None
pending = [candidate('baseline')]
results = []
if os.environ.get('FOUNTAIN_RESUME')=='1' and (ROOT/'latest.json').exists():
    previous=json.loads((ROOT/'latest.json').read_text())
    results=previous['results']
    pending=[]
    state['resumed_from']=previous['run']
mode = 'water'
vessels_frosted = True


def json_write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def material(path,color,opacity=1.,roughness=.3,metal=0.):
    mat=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    for k,typ,val in [('diffuseColor',Sdf.ValueTypeNames.Color3f,Gf.Vec3f(*color)),
                      ('roughness',Sdf.ValueTypeNames.Float,roughness),
                      ('metallic',Sdf.ValueTypeNames.Float,metal),
                      ('opacity',Sdf.ValueTypeNames.Float,opacity),
                      ('ior',Sdf.ValueTypeNames.Float,1.333)]:
        shader.CreateInput(k,typ).Set(val)
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return mat


def cube(name,pos,size,mat,collision=True,parent='/World'):
    obj=UsdGeom.Cube.Define(stage,parent+'/'+name)
    obj.CreateSizeAttr(1.)
    obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
    obj.AddScaleOp().Set(Gf.Vec3d(*size))
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        api=PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
        api.CreateContactOffsetAttr(.001)
        api.CreateRestOffsetAttr(0.)
        if stage.GetPrimAtPath('/World/WetContact'):
            physicsUtils.add_physics_material_to_prim(stage,obj.GetPrim(),Sdf.Path('/World/WetContact'))
    return obj


def set_view(close=False):
    if (state.get('trial') or {}).get('robot'):
        eye = [.94, -.66, 1.36] if close else [1.50, -1.00, 1.74]
        target = [.49, -.19, 1.11] if close else [.30, -.12, 1.25]
    else:
        eye = [.22,-.30,.22] if close else [.32,-.44,.30]
        target = [0.,0.,.045] if close else [0.,0.,.10]
    set_camera_view(eye=np.array(eye), target=np.array(target))
    cam=UsdGeom.Camera(stage.GetPrimAtPath('/OmniverseKit_Persp'))
    if cam:
        cam.CreateClippingRangeAttr(Gf.Vec2f(.001,20.))
        cam.CreateFocalLengthAttr(45.)


def water_color():
    if not water: return
    shader=UsdShade.Shader(stage.GetPrimAtPath('/World/Water/Shader'))
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(*((.14,.35,.45) if mode=='water' else (.25,.012,.025))))
    # Same physical parameters; wine here changes only the display material.


def display_look(look,spec):
    if not water: return
    water.GetPrim().RemoveProperty('outputs:mdl:surface')
    shader=UsdShade.Shader(stage.GetPrimAtPath('/World/Water/Shader'))
    shader.GetInput('opacity').Set({'clearer':.07,'visible':.25}.get(look,spec.get('opacity',.14)))
    shader.GetInput('roughness').Set({'clearer':.025,'visible':.04}.get(look,spec.get('roughness',.06)))
    if look=='mdl':
        mdl=UsdShade.Shader.Define(stage,'/World/Water/MDL')
        mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        mdl.SetSourceAsset(Sdf.AssetPath('/isaac-sim/kit/mdl/core/Base/OmniSurfacePresets.mdl'),'mdl')
        mdl.SetSourceAssetSubIdentifier('OmniSurface_ClearWater','mdl')
        mdl.CreateOutput('out',Sdf.ValueTypeNames.Token)
        water.CreateSurfaceOutput('mdl').ConnectToSource(mdl.ConnectableAPI(),'out')
    state['display_look']=look


def fountain_fixture(spec,dark,ceramic,glass,chrome):
    cube('Bench',(0,0,-.018),(.36,.30,.012),dark)
    receiver_orient=None
    if spec.get('hinge'):
        receiver=UsdGeom.Xform.Define(stage,'/World/Receiver')
        receiver.AddTranslateOp().Set(Gf.Vec3d(*PIVOT))
        receiver_orient=receiver.AddOrientOp()
        receiver_orient.Set(Gf.Quatf(1.,Gf.Vec3f(0.)))
        rigid=UsdPhysics.RigidBodyAPI.Apply(receiver.GetPrim())
        rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(receiver.GetPrim()).CreateMassAttr(.3)
        axle=UsdGeom.Cylinder.Define(stage,'/World/HingeAxle')
        axle.CreateAxisAttr('Y');axle.CreateRadiusAttr(.004);axle.CreateHeightAttr(.114)
        axle.AddTranslateOp().Set(Gf.Vec3d(-.055,0.,-.009))
        UsdShade.MaterialBindingAPI.Apply(axle.GetPrim()).Bind(chrome)
    def receiver_cube(name,pos,size,mat,collision=True):
        if spec.get('hinge'):
            return cube(name,np.asarray(pos)-PIVOT,size,mat,collision,parent='/World/Receiver')
        return cube(name,pos,size,mat,collision)
    receiver_cube('Floor',(0,0,-.006),(.11,.09,.012),ceramic,spec['collide'])
    for name,pos,size in [('Left',(-.0525,0,WALL_HEIGHT/2),(.005,.09,WALL_HEIGHT)),
                           ('Right',(.0525,0,WALL_HEIGHT/2),(.005,.09,WALL_HEIGHT)),
                           ('Back',(0,.0425,WALL_HEIGHT/2),(.10,.005,WALL_HEIGHT)),
                           ('Front',(0,-.0425,WALL_HEIGHT/2),(.10,.005,WALL_HEIGHT))]:
        receiver_cube(name,pos,size,glass,spec['collide'])
    # Non-colliding floor tiles make surface transmission and leveling visible.
    for x in range(10):
        for y in range(8):
            if (x+y)%2:
                receiver_cube(f'Tile{x}_{y}',(-.045+x*.01,-.035+y*.01,.00005),(.01,.01,.0001),dark,False)
    if spec.get('obstacles'):
        colors=[(.9,.39,.07),(.055,.43,.62)]
        for obstacle,color in zip(OBSTACLES,colors):
            mat=material('/World/'+obstacle['name']+'Material',color,roughness=.24)
            receiver_cube(obstacle['name'],obstacle['center'],obstacle['size'],mat,True)
    nozzle=UsdGeom.Cylinder.Define(stage,'/World/Nozzle')
    nozzle.CreateRadiusAttr(.007);nozzle.CreateHeightAttr(.025)
    nozzle.AddTranslateOp().Set(Gf.Vec3d(0,0,spec['height']+.0125))
    UsdShade.MaterialBindingAPI.Apply(nozzle.GetPrim()).Bind(chrome)
    return receiver_orient


def setup(spec):
    global sim,stage,water
    if current and current.get('recording'):
        current['recording'].abort('Scene replaced')
    if sim:
        sim.stop()
        SimulationContext.clear_instance()
    create_new_stage()
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,'/World')
    sim=SimulationContext(physics_dt=spec['dt'],rendering_dt=.05,stage_units_in_meters=1.,
                          physics_prim_path='/World/Physics',backend='numpy',device='cpu')
    physics=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/World/Physics'))
    physics.CreateEnableGPUDynamicsAttr(True)
    physics.CreateBroadphaseTypeAttr('GPU')
    physics.CreateEnableCCDAttr(True)
    physics.CreateTimeStepsPerSecondAttr(round(1/spec['dt']))
    scene=UsdPhysics.Scene(stage.GetPrimAtPath('/World/Physics'))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1)); scene.CreateGravityMagnitudeAttr(9.81)
    if spec.get('wall_friction') is not None:
        mat=UsdShade.Material.Define(stage,'/World/WetContact')
        api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(spec['wall_friction'])
        api.CreateDynamicFrictionAttr(spec['wall_friction'])
        api.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('min')
    dark=material('/World/Dark',(.085,.11,.14),roughness=.24)
    ceramic=material('/World/Ceramic',(.67,.72,.74),roughness=.22)
    glass=material('/World/Glass',(.55,.7,.75),opacity=.10,roughness=.04)
    chrome=material('/World/Chrome',(.62,.65,.68),roughness=.15,metal=.9)
    water=material('/World/Water',(.14,.35,.45),opacity=spec['opacity'],roughness=spec['roughness'])
    water_color()
    display_look(spec.get('appearance','default'),spec)
    bottle=None
    receiver_orient=None
    if spec.get('bottle'):
        shader=UsdShade.Shader(stage.GetPrimAtPath('/World/Water/Shader'))
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(.62,.012,.052))
        cup_glass=material('/World/ClearTumbler',(.80,.91,.94),opacity=.055,roughness=.035)
        bottle_glass=material('/World/ClearBottle',(.77,.91,.82),opacity=.07,roughness=.035)
        bench=material('/World/Counter',(.55,.54,.49),roughness=.48)
        if spec.get('robot'):
            from robot_pour import RobotPour, CUP_WORLD
            cube('Bench',CUP_WORLD+[.005,-.05,-.012],(.38,.46,.024),bench)
            bottle=RobotPour(stage,spec,bottle_glass,cup_glass)
        else:
            cube('Bench',(-.04,0,-.006),(.46,.29,.012),bench)
            bottle=Pour(stage,spec,bottle_glass,cup_glass)
        from vessel_appearance import apply as apply_vessel_appearance
        apply_vessel_appearance(stage, True if spec.get('record_cameras') else vessels_frosted, variant='A')
        state['vessels_frosted'] = True if spec.get('record_cameras') else vessels_frosted
    else:
        receiver_orient=fountain_fixture(spec,dark,ceramic,glass,chrome)
    dome=UsdLux.DomeLight.Define(stage,'/World/Dome');dome.CreateIntensityAttr(700.)
    for i,pos in enumerate(((.1,-.1,.45),(-.15,.15,.35))):
        light=UsdLux.RectLight.Define(stage,f'/World/Light{i}')
        light.CreateWidthAttr(.25);light.CreateHeightAttr(.15);light.CreateIntensityAttr(1600.)
        light.AddTranslateOp().Set(Gf.Vec3d(*pos))
    s=spec['spacing'];contact=s/(2*.99*.6)*spec['contact_ratio']
    system=Sdf.Path('/World/FluidSystem')
    particleUtils.add_physx_particle_system(stage,system,contact_offset=contact,rest_offset=contact*.99,
        particle_contact_offset=contact,solid_rest_offset=contact*.99,fluid_rest_offset=s/2,
        solver_position_iterations=spec['iterations'],simulation_owner='/World/Physics',max_neighborhood=spec['neighbors'],
        enable_ccd=True,max_depenetration_velocity=1.,max_velocity=6.,
        global_self_collision_enabled=True,non_particle_collision_enabled=True)
    physicsmat=Sdf.Path('/World/FluidPhysics')
    particleUtils.add_pbd_particle_material(stage,physicsmat,viscosity=spec['viscosity'],cohesion=spec['cohesion'],
        surface_tension=spec['surface_tension'],friction=spec['friction'],adhesion=spec['adhesion'],
        damping=spec['damping'],vorticity_confinement=spec['vorticity'])
    physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system),physicsmat)
    particleUtils.add_physx_particle_isosurface(stage,system,enabled=True,max_vertices=spec.get('isosurface_max_vertices',131072),max_triangles=spec.get('isosurface_max_triangles',262144),
        max_subgrids=spec.get('isosurface_max_subgrids',8192 if s<=.0012 else 4096),grid_spacing=s*spec['grid_spacing'],surface_distance=s*spec['surface_distance'],
        grid_filtering_passes='GS',grid_smoothing_radius=s*spec['smoothing'],
        num_mesh_smoothing_passes=spec['mesh_passes'],num_mesh_normal_smoothing_passes=4)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(system)).Bind(water)
    # Rotate the sampling lattice between repeats, without changing flow or volume.
    rng=np.random.default_rng(spec['seed'])
    angle=rng.uniform(0,2*np.pi)
    limit=math.ceil(.005/s)
    cross=np.array([(x*s,y*s) for x in range(-limit,limit+1) for y in range(-limit,limit+1)
                    if (x*s)**2+(y*s)**2 <= .005**2])
    cross=cross@np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    period=len(cross)*s**3/(spec['flow_ml_s']*1e-6)
    speed=s/period
    points=np.column_stack((cross,np.full(len(cross),spec['height']))).astype(np.float32)
    velocities=np.tile([0.,0.,-speed],(len(points),1)).astype(np.float32)
    if bottle is not None:
        points=bottle.points
        velocities=np.zeros_like(points)
    prim=particleUtils.add_physx_particleset_pointinstancer(stage,Sdf.Path('/World/Particles'),
        Vt.Vec3fArray.FromNumpy(points),Vt.Vec3fArray.FromNumpy(velocities),system,
        self_collision=True,fluid=True,particle_group=1,
        particle_mass=0. if spec.get('mass_mode')=='density' else 1000*s**3,
        density=1000. if spec.get('mass_mode')=='density' else 0.,num_prototypes=1)
    inst=UsdGeom.PointInstancer(prim)
    prototype=UsdGeom.Sphere(stage.GetPrimAtPath('/World/Particles/particlePrototype0'))
    prototype.CreateRadiusAttr(s/2);prototype.CreateVisibilityAttr('invisible')
    context=sim.get_physics_context()
    direct=type(context)._step.__get__(context,type(context))
    interface=omni.physx.get_physx_interface()
    sub=[0];interval=round(.05/spec['dt'])
    def step(current_time,update_fabric=False):
        sub[0]+=1
        if sub[0]%interval==0:
            interface.update_simulation(context.get_physics_dt(),current_time)
            interface.update_transformations(False,True,True,False)
        else: direct(current_time,update_fabric)
    context._step=step
    set_view(close=bool(spec.get('hinge')))
    if bottle is not None:
        set_camera_view(eye=np.array([.12,-.39,.19]),target=np.array([-.055,0,.082]))
    if spec.get('robot'):
        set_view()
        for i in range(2):
            op=UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/Light{i}')).GetOrderedXformOps()[0]
            op.Set(Gf.Vec3d(*(np.asarray(op.Get())+CUP_WORLD)))
    sim.reset()
    state['ready']=True
    folder=OUT/f'{len(results)+1:03}-{spec["name"]}-{time.time_ns()%1000000}'
    (folder/'frames').mkdir(parents=True)
    if spec.get('bottle'):
        spec['vessel_appearance'] = 'A' if spec.get('record_cameras') or vessels_frosted else 'clear'
    recording = None
    if spec.get('record_cameras'):
        from pour_recording import PourRecording
        recording = PourRecording(stage, folder, spec, sim, bottle, preview)
        state['recording'] = dict(active=True, path=str(folder/'episode.partial.hdf5'),
                                  cameras=list(recording.cameras), resolution=[384,384],
                                  camera_resolutions={k:list(v) for k,v in recording.camera_sizes.items()}, control_hz=20, frames=0)
    actor = None
    if spec.get('policy_evaluation'):
        if spec.get('policy_family')=='visual_guard_hierarchy':
            from pour_guard_actor import Actor
        elif spec.get('policy_family')=='visual_stop_hierarchy':
            from pour_stop_actor import Actor
        else:
            from pour_learning import Actor
        actor=Actor(spec['checkpoint'])
        if 'cup_height_scale' in spec or 'cup_diameter_scale' in spec:
            from pour_vessel_geometry import cup_geometry
            declared=cup_geometry(spec)
            assert actor.saved['calibration'].get('cup_geometry')==declared, 'Checkpoint does not support this declared cup geometry'
            assert actor.saved.get('size_curriculum_native_accepted'), 'Size-aware autonomous pouring requires calibration and clearance acceptance first'
    replay = None
    if spec.get('replay_episode'):
        import sys
        sys.path.append('/workspace/pydeps51')
        import h5py
        with h5py.File(spec['replay_episode'], 'r') as saved:
            replay = dict(actions=np.degrees(saved['data/demo_0/actions'][:]),
                          angles=saved['data/demo_0/ground_truth/command_angle_deg'][:])
    json_write(folder/'parameters.json',spec)
    if bottle is not None and spec.get('save_scene',True):
        if spec.get('robot'):
            stage.GetRootLayer().Export(str(folder/'scene-start.usda'))
        else:
            stage.Export(str(folder/'scene-start.usda'))
    if spec.get('hinge'):
        json_write(folder/'container.json',dict(pivot_m=PIVOT.tolist(),axis='negative Y for positive lift',
            kind='prescribed kinematic hinge; accelerating return and hard stop at table',
            obstacles=OBSTACLES,fluid_impulse=False,table_impact_force_measured=False))
        stage.Export(str(folder/'scene-start.usda'))
    return dict(spec=spec,bottle=bottle,inst=inst,cross=cross,period=period,speed=speed,emitted_layers=1,
                expected_count=len(points),t=float(sim.current_time) if recording else 0.,recording=recording,replay=replay,samples=[],folder=folder,frame=0,wall=time.monotonic(),
                actor=actor,trajectory=[],mesh_counts=[],kicked=False,receiver_orient=receiver_orient,
                readback_audit=dict(calls=0,max_position_correction_m=0.,max_velocity_correction_m_s=0.))


def read_particles(trial):
    inst=trial['inst']
    return np.asarray(inst.GetPositionsAttr().Get(),dtype=np.float32).copy(),np.asarray(inst.GetVelocitiesAttr().Get(),dtype=np.float32).copy()


def emit(trial):
    spec=trial['spec']; t=trial['t']; period=trial['period']
    last=min(t,spec['flow_seconds']-1e-9)
    layers=math.floor(last/period)+1
    if layers<=trial['emitted_layers']: return
    old_p,old_v=read_particles(trial)
    # fetch_results alone does not guarantee the USD particle arrays are current.
    # Flush physical readback before appending; otherwise emission can rewind the
    # existing fluid to the last rendered frame.
    omni.physx.get_physx_interface().update_transformations(False,True,True,False)
    p,v=read_particles(trial)
    audit=trial['readback_audit'];audit['calls']+=1
    if len(old_p)==len(p) and len(p):
        audit['max_position_correction_m']=max(audit['max_position_correction_m'],float(np.max(np.linalg.norm(p-old_p,axis=1))))
        audit['max_velocity_correction_m_s']=max(audit['max_velocity_correction_m_s'],float(np.max(np.linalg.norm(v-old_v,axis=1))))
    if len(p) != trial['expected_count']:
        raise RuntimeError('Physical particle count changed before emission')
    additions=[];vels=[]
    for i in range(trial['emitted_layers'],layers):
        age=t-i*period
        xy=trial['cross'].copy()
        # Smooth sweep from center to off-center in the second half.
        f=np.clip((i*period-1.25)/.5,0,1)
        xy[:,0]+=spec['offset_x']*f
        z=spec['height']-trial['speed']*age-.5*9.81*age*age
        additions.append(np.column_stack((xy,np.full(len(xy),z))))
        vels.append(np.tile([0.,0.,-trial['speed']-9.81*age],(len(xy),1)))
    p=np.concatenate([p,*additions]).astype(np.float32)
    v=np.concatenate([v,*vels]).astype(np.float32)
    if len(p)>80000: raise RuntimeError('Particle budget exceeded')
    inst=trial['inst']
    with Sdf.ChangeBlock():
        inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p))
        inst.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(v))
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0]*len(p)))
        # USD stores set mass, so maintain constant particle mass while emitting.
        if spec.get('mass_mode')!='density':
            UsdPhysics.MassAPI(inst.GetPrim()).GetMassAttr().Set(len(p)*1000*spec['spacing']**3)
    trial['emitted_layers']=layers;trial['expected_count']=layers*len(trial['cross'])
    x=spec['offset_x']*np.clip((t-1.25)/.5,0,1)
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Nozzle')).GetOrderedXformOps()[0].Set(Gf.Vec3d(float(x),0,spec['height']+.0125))


def best_result():
    valid=[r for r in results if r['spec'].get('protocol')==3 and r['metrics']['eligible'] and not r['spec']['name'].startswith(('validate_','render_','replay_'))]
    discriminated=[r for r in valid if 'water_screening_score' in r['metrics']]
    if discriminated: return max(discriminated,key=lambda r:r['metrics']['water_screening_score'])
    return max(valid,key=lambda r:r['metrics']['screening_score']) if valid else None


def selected_result():
    path=ROOT/'selected-preset.json'
    if not path.exists(): return None
    selection=json.loads(path.read_text())
    # Physical settings must come from a completed eligible trial. Display-only
    # overrides are restricted to the four built-in, independently reviewed looks.
    chosen=next((r for r in results if r['path']==selection.get('source_path') and r['metrics']['eligible']),None)
    if chosen and selection.get('appearance') in {'default','mdl','clearer','visible'}:
        chosen=dict(chosen,spec=dict(chosen['spec'],appearance=selection['appearance']))
    return chosen


def audit_scene():
    from scipy.spatial import cKDTree
    folder=ROOT/'scene-audit';folder.mkdir(exist_ok=True)
    inst=UsdGeom.PointInstancer(stage.GetPrimAtPath('/World/Particles'))
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/World/FluidSystem/Isosurface'))
    points=np.asarray(inst.GetPositionsAttr().Get())
    vertices=np.asarray(mesh.GetPointsAttr().Get())
    indices=np.unique(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()))
    used=vertices[indices]
    distances=cKDTree(points).query(used)[0]
    bindings={}
    for path in ('/World/Particles','/World/Particles/particlePrototype0','/World/FluidSystem','/World/FluidSystem/Isosurface'):
        prim=stage.GetPrimAtPath(path)
        mat=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        bindings[path]=dict(visibility=str(UsdGeom.Imageable(prim).ComputeVisibility()),material=str(mat.GetPath()) if mat else None,
                            apis=list(prim.GetAppliedSchemas()))
    physical_attrs={str(a.GetName()):str(a.Get()) for a in inst.GetPrim().GetAttributes() if str(a.GetName()).startswith(('physxParticle','physics:'))}
    json_write(folder/'audit.json',dict(sim_time=state.get('sim_time'),look=state.get('display_look'),bindings=bindings,
        particle_attributes=physical_attrs,particles=len(points),mesh_vertices=len(vertices),used_vertices=len(used),
        mesh_to_particle_p99_m=float(np.quantile(distances,.99)),mesh_to_particle_max_m=float(distances.max()),
        particle_z_quantiles=np.quantile(points[:,2],[0,.5,.9,.99,1]).tolist()))
    np.savez_compressed(folder/'geometry.npz',particles=points,mesh_vertices=used)
    stage.Export(str(folder/'scene.usda'))


def save_results():
    state['results']=[dict(name=r['spec']['name'],trial_index=i,**r['metrics']) for i,r in enumerate(results) if r['spec'].get('protocol') in (3,4)]
    allowed_commands.update(f'trial_{i}' for i in range(len(results)))
    state['archived_trials']=sum(r['spec'].get('protocol') not in (3,4) for r in results)
    best=best_result()
    state['best']=None if not best else dict(name=best['spec']['name'],score=best['metrics'].get('water_screening_score',best['metrics']['screening_score']))
    selected=selected_result()
    state['selected']=None if not selected else selected['spec']['name']
    json_write(OUT/'results.json',results)
    json_write(ROOT/'latest.json',dict(run=RUN,path=str(OUT),results=results,best=best))
    if best: json_write(ROOT/'best-tested.json',best)


def finish(trial):
    if trial['bottle'] is None:
        metrics=summarize(trial['samples'],trial['spec'])
    else:
        metrics=trial['bottle'].summary(trial['samples'])
        state['bottle_result']=metrics

    if trial['spec'].get('hinge'):
        metrics['hinge']=hinge_response(trial['samples'],trial['spec'])
        state['hinge_result']=metrics['hinge']
        metrics['comparison_group']='hinged container with obstacles; separate from fountain parameter ranking'
        if any(metrics['hinge']['peak_obstacle_penetration']):
            metrics['gates'].append('obstacle_penetration');metrics['eligible']=False
    if trial.get('actor'):
        actor=trial['actor']
        metrics['policy']=dict(checkpoint=trial['spec']['checkpoint'],steps=actor.steps,
                               checkpoint_sha256=actor.checkpoint_sha256,
                               checkpoint_update=actor.saved['step'],
                               model_version=actor.saved.get('model_version','residual_bc_v1'),
                               bounded_action_steps=actor.clamps,done=actor.complete(),
                               inputs=actor.saved['inputs'],teacher_fallback=False)
        if not actor.complete() and not trial['spec'].get('expert_recovery'):
            metrics['gates'].append('policy_did_not_finish')
            metrics['eligible']=False
            metrics['task_success_zero_spill']=False
            if 'surface_task_success_zero_spill' in metrics:metrics['surface_task_success_zero_spill']=False
        if trial.get('evaluation_termination'):
            metrics['gates'].append(trial['evaluation_termination'])
            metrics['eligible']=False
            metrics['task_success_zero_spill']=False
            if 'surface_task_success_zero_spill' in metrics:metrics['surface_task_success_zero_spill']=False
        json_write(trial['folder']/'policy-decisions.json',actor.diagnostics)
        if trial['spec'].get('expert_recovery'):
            expert=trial.get('expert')
            metrics['recovery']=dict(started=expert is not None,
                teacher_complete=bool(expert and expert.complete(trial['t'])),
                prefix_checkpoint_sha256=actor.checkpoint_sha256,
                evidence=expert.evidence if expert else None,
                planning_error=trial.get('recovery_error'))
            metrics['policy']['teacher_fallback']=True
            metrics['policy']['autonomous_task_success']=False
            if not expert or not expert.complete(trial['t']):
                metrics['gates'].append('expert_recovery_did_not_finish')
                metrics['eligible']=False
            metrics['recovery_success_zero_spill']=bool(metrics['eligible'] and metrics['zero_spill'] and
                (metrics['expected_stop']!='cup_near_full' or metrics['bottle_result']['cup_inventory_fraction']>=.90))
            metrics['task_success_zero_spill']=False
    if trial.get('recording'):
        recording_cameras = list(trial['recording'].cameras)
        recording_path = trial['recording'].finish(metrics)
        metrics['recording'] = str(recording_path)
        if trial['spec'].get('replay_episode'):
            metrics['replay_source'] = trial['spec']['replay_episode']
        elif metrics['eligible'] and not trial['spec'].get('policy_evaluation') and not trial['spec'].get('corrective_prefix'):
            json_write(ROOT/'latest-recording.json', dict(path=str(recording_path), spec=trial['spec']))

        state['recording'] = dict(active=False, path=str(recording_path),
                                  cameras=recording_cameras, resolution=[384,384], control_hz=20,
                                  camera_resolutions={k:list(v) for k,v in trial['recording'].camera_sizes.items()},
                                  frames=trial['recording'].count)
    metrics['wall_seconds']=round(time.monotonic()-trial['wall'],2)
    result=dict(spec=trial['spec'],metrics=metrics,path=str(trial['folder']))
    json_write(trial['folder']/'metrics.json',metrics)
    json_write(trial['folder']/'readback-audit.json',trial['readback_audit'])
    json_write(trial['folder']/'samples.json',trial['samples'])
    if trial['trajectory']:
        arrays={f'positions_{i:03}':p for i,(t,p,v) in enumerate(trial['trajectory'])}
        arrays.update({f'velocities_{i:03}':v for i,(t,p,v) in enumerate(trial['trajectory'])})
        arrays['times']=np.array([t for t,p,v in trial['trajectory']])
        np.savez_compressed(trial['folder']/'trajectory.npz',**arrays)
    results.append(result);save_results()
    print('FOUNTAIN_RESULT '+json.dumps(result),flush=True)


state['phase']='Ready'
json_write(OUT/'experiment.json',dict(version=1,plan=coarse_plan(),geometry=dict(inner_width=.10,inner_depth=.08,floor_thickness=.012,wall_height=WALL_HEIGHT),
    phases=['center jet 0–1.25 s','move nozzle 1.25–1.75 s','off-center jet to 3 s','flow off; settle 3–6 s'],
    caveats=['Native PBD coefficients, not SI viscosity','No real-water reference or automatic visual model calibration',
             'Geometric particle neighbor and wall proximity proxies, not contact impulses'],gpu_uuid=GPU))
print('FOUNTAIN_APP_READY',flush=True)
save_results()
try:
    while app.is_running():
        try:
            while True:
                cmd=preview.commands.get_nowait()
                if (current and current.get('recording') and cmd not in
                        {'pause','stop','robot_full','robot_empty','robot_replay','close_view','wide_view'}):
                    state['control_message'] = 'Finish the recording before changing the scene appearance.'
                    continue
                if cmd=='pause': state['paused']=not state['paused']
                elif cmd=='stop':
                    pending=[];state['paused']=True;state['phase']='Stopped by user'
                    if current and current.get('recording'):
                        current['recording'].abort('Stopped by user')
                        state['recording']['active'] = False
                        current = None
                elif cmd == 'vessels_review':
                    if current is not None or sim is not None:
                        raise RuntimeError('Material review requires a fresh renderer')
                    saved = json.loads(Path('/workspace/light-frost-review-state.json').read_text())
                    omni.usd.get_context().open_stage(str(ROOT/'scene-audit'/'scene.usda'))
                    stage = omni.usd.get_context().get_stage()
                    water = UsdShade.Material.Get(stage, '/World/Water')
                    state.update(trial=dict(saved['trial'], material_review=True),
                                 metrics=saved['metrics'], sim_time=saved['sim_time'],
                                 phase='Material preview', ready=True, progress=1.)
                    UsdGeom.Imageable(stage.GetPrimAtPath('/World/Robot')).CreateVisibilityAttr().Set('invisible')
                    # Frozen half-height fill makes the vessel waterline readable in still previews.
                    # This geometry edit is for material inspection with the timeline stopped.
                    surface = UsdGeom.Mesh(stage.GetPrimAtPath('/World/FluidSystem/Isosurface'))
                    vertices = np.asarray(surface.GetPointsAttr().Get(), dtype=np.float32).copy()
                    cup_center = np.array([.49, -.15])
                    mask = (np.linalg.norm(vertices[:,:2]-cup_center,axis=1)<.03) & (vertices[:,2]>1.043)
                    old_height = vertices[mask,2]-1.04
                    new_height = .004+(old_height-.004)*.48
                    old_radius = .020+(old_height-.004)/.034*.004
                    new_radius = .020+(new_height-.004)/.034*.004
                    vertices[mask,:2] = cup_center+(vertices[mask,:2]-cup_center)*((new_radius-.001)/(old_radius-.001))[:,None]
                    vertices[mask,2] = 1.04+new_height
                    surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
                    settings.set('/app/viewport/grid/enabled', False)
                    liquid_shader = UsdShade.Shader(stage.GetPrimAtPath('/World/Water/Shader'))
                    liquid_shader.GetInput('opacity').Set(1.)
                    liquid_shader.GetInput('roughness').Set(.22)
                    state['metrics'] = {}
                    state['review_note'] = 'Static appearance comparison with a half-height cup fill mesh. No physical-volume claim.'

                    set_camera_view(eye=np.array([.91,-.55,1.19]), target=np.array([.49,-.21,1.10]))
                    from vessel_appearance import apply as apply_vessel_appearance
                    apply_vessel_appearance(stage, True, variant='B')
                    state.update(vessels_frosted=True, vessel_review_variant='B')
                elif cmd in ('props_a','props_b','props_c'):
                    if stage and stage.GetPrimAtPath('/World/Bottle'):
                        from vessel_appearance import apply as apply_vessel_appearance
                        variant = cmd[-1].upper()
                        apply_vessel_appearance(stage, True, variant=variant)
                        state.update(vessels_frosted=True, vessel_review_variant=variant)
                elif cmd in ('props_frosted','props_clear'):
                    vessels_frosted = cmd == 'props_frosted'
                    if stage and stage.GetPrimAtPath('/World/Bottle'):
                        from vessel_appearance import apply as apply_vessel_appearance
                        apply_vessel_appearance(stage, vessels_frosted)
                    state['vessels_frosted'] = vessels_frosted
                elif cmd in ('water','wine'):
                    mode=cmd;state['appearance']=mode
                    if cmd=='wine' and water: display_look('default',state.get('trial') or BASE)
                    water_color()
                elif cmd.startswith('look_'):
                    display_look(cmd.removeprefix('look_'),state.get('trial') or BASE)
                elif cmd in ('hide_particles','show_particles') and stage:
                    UsdGeom.Imageable(stage.GetPrimAtPath('/World/Particles')).GetVisibilityAttr().Set('invisible' if cmd=='hide_particles' else 'inherited')
                    state['particle_display']=cmd
                elif cmd=='audit_scene' and stage: audit_scene()
                elif cmd.startswith('trial_'):
                    index=int(cmd.removeprefix('trial_'))
                    if 0<=index<len(results):
                        pending.append(dict(results[index]['spec'],name='replay_'+results[index]['spec']['name']))
                        pending=pending[:32];state['paused']=False
                elif cmd in ('close_view','wide_view'):
                    if stage: set_view(cmd=='close_view')
                else:
                    best=best_result()
                    if cmd in ('perception_pilot','perception_collect','perception_geometry','perception_camera_pilot','perception_holdout','perception_development','perception_test'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before perception-study work';continue
                        from pour_perception_study import pending as perception_pending
                        pending.extend(perception_pending(cmd.removeprefix('perception_')))
                    elif cmd in ('size_calibration','size_seed_calibration','size_recovery_development'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before size-curriculum work';continue
                        from pour_size_study import pending as size_pending
                        pending.extend(size_pending({'size_calibration':'calibration','size_seed_calibration':'calibration_seed',
                            'size_recovery_development':'recovery_development'}[cmd]))
                    elif cmd in ('finish_train','finish_calibrate','finish_calibration_pilot','finish_development','finish_test'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before finish-study work';continue
                        if cmd=='finish_train':
                            try:
                                from pour_finish_train import train
                                state['trained_checkpoint']=train(state)
                                state.pop('learning_error',None)
                            except Exception:
                                state['learning_error']=traceback.format_exc();state['phase']='Finish-study fitting failed'
                        else:
                            from pour_finish_study import calibrations,development,tests
                            pending.extend(calibrations()[:3] if cmd=='finish_calibration_pilot' else calibrations() if cmd=='finish_calibrate' else development() if cmd=='finish_development' else tests())
                    elif cmd in ('wide_pilot','wide_repair','wide_collect','wide_development','wide_test'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing wider-table cases';continue
                        from pour_wide_study import references,repairs,development,tests
                        pending.extend(references(4) if cmd=='wide_pilot' else
                            repairs() if cmd=='wide_repair' else
                            references() if cmd=='wide_collect' else development() if cmd=='wide_development' else tests())
                    elif cmd=='wide_train':
                        if current is not None or pending:
                            state['control_message']='Wait for idle before fitting wider-table models';continue
                        try:
                            from pour_wide_train import train
                            state['trained_checkpoint']=train(state)
                            state.pop('learning_error',None)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Wider-table fitting failed'
                    elif cmd=='layout_variety_extra':
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing additional layouts';continue
                        from pour_layout_extra import unstarted
                        pending.extend(unstarted())
                    elif cmd=='layout_variety_test':
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing layout variety tests';continue
                        from pour_layout_variety import unstarted
                        pending.extend(unstarted())
                    elif cmd=='workspace_train':
                        if current is not None or pending:
                            state['control_message']='Wait for active trials to finish';continue
                        try:
                            from pour_workspace_train import train
                            state['trained_checkpoint']=train(state)
                            state.pop('learning_error',None)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Expanded-workspace fitting failed'
                        continue
                    elif cmd in ('workspace_collect','workspace_development','workspace_test'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing expanded-workspace cases';continue
                        from pour_workspace_study import demonstrations,development,unstarted,ROOT as WORKSPACE_ROOT
                        if cmd!='workspace_collect' and not (WORKSPACE_ROOT/'best.pt').exists():
                            state['control_message']='Fit expanded-workspace model first';continue
                        pending.extend({'workspace_collect':demonstrations,'workspace_development':development,'workspace_test':unstarted}[cmd]())
                    elif cmd=='reliable_train':
                        if current is not None or pending:
                            state['control_message']='Wait for active trials to finish';continue
                        try:
                            from pour_reliable_train import train
                            state['trained_checkpoint']=train(state)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Reliable-pour fitting failed'
                        continue
                    elif cmd in ('reliable_development','reliable_test'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing reliable-pour cases';continue
                        from pour_reliable_study import development,unstarted,ROOT as RELIABLE_ROOT
                        if not (RELIABLE_ROOT/'best.pt').exists():
                            state['control_message']='Fit reliable-pour predictor first';continue
                        pending.extend(development() if cmd=='reliable_development' else unstarted())
                    elif cmd=='settled_train':
                        if current is not None or pending:
                            state['control_message']='Wait for active trials to finish';continue
                        try:
                            from pour_settled_train import train
                            state['trained_checkpoint']=train(state)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Settled-height fitting failed'
                        continue
                    elif cmd in ('settled_development','settled_test'):
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing settled-height cases';continue
                        from pour_settled_study import development,unstarted,ROOT as SETTLED_ROOT
                        if not (SETTLED_ROOT/'best.pt').exists():
                            state['control_message']='Fit settled-height predictor first';continue
                        pending.extend(development() if cmd=='settled_development' else unstarted())
                    elif cmd=='guard_train':
                        if current is not None or pending:
                            state['control_message']='Wait for active trials to finish';continue
                        try:
                            from pour_guard_train import train
                            state['trained_checkpoint']=train(state)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Visual-guard training failed'
                        continue
                    elif cmd=='robust_test':
                        if current is not None or pending:
                            state['control_message']='Wait for idle before queuing robustness cases';continue
                        from pour_robust_study import unstarted
                        pending.extend(unstarted())
                    elif cmd in ('vote_development','vote_test'):
                        from pour_guard_vote_study import plan as vote_plan,ROOT as VOTE_ROOT
                        if not (VOTE_ROOT/'best.pt').exists():
                            state['control_message']='Select the calibrated confirmation policy first';continue
                        frozen=vote_plan()
                        pending.extend(frozen['regression']+frozen['validation'] if cmd=='vote_development' else frozen['test'])
                    elif cmd in ('guard_val','guard_regression','guard_development','guard_test'):
                        from pour_guard_study import plan as guard_plan,ROOT as GUARD_ROOT
                        if not (GUARD_ROOT/'best.pt').exists():
                            state['control_message']='Fit visual guard policy first';continue
                        frozen=guard_plan()
                        if cmd=='guard_development':pending.extend(frozen['regression']+frozen['validation'])
                        else:
                            part={'guard_val':'validation','guard_regression':'regression','guard_test':'test'}[cmd]
                            pending.extend(frozen[part])
                    elif cmd=='stop_train':
                        if current is not None or pending:
                            state['control_message']='Wait for active trials to finish';continue
                        try:
                            from pour_stop_train import train
                            state['trained_checkpoint']=train(state)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Visual-stop training failed'
                        continue
                    elif cmd in ('stop_val','stop_test'):
                        from pour_stop_study import plan as stop_plan,ROOT as STOP_ROOT
                        if not (STOP_ROOT/'best.pt').exists():
                            state['control_message']='Fit the visual-stop policy first';continue
                        part='validation' if cmd=='stop_val' else 'test'
                        pending.extend(stop_plan()[part])
                    elif cmd in ('recovery_collect','recovery_probe'):
                        from pour_recovery_study import plan as recovery_plan
                        study=recovery_plan((selected_result() or best)['spec'])
                        seen={r['spec']['name'] for r in results}
                        seen.update(s['name'] for s in pending)
                        candidates=[s for s in study['train']+study['validation']
                            if s.get('expert_recovery') and s['name'] not in seen]
                        pending.extend(candidates[:2] if cmd=='recovery_probe' else candidates)
                    elif cmd in ('recovery_train','release_train'):
                        if current is not None or pending:
                            state['control_message']='Wait for recovery collection to finish'
                            continue
                        try:
                            if cmd=='release_train':
                                from pour_release_learning import train
                            else:
                                from pour_recovery_learning import train
                            state['trained_checkpoint']=train(state)
                        except Exception:
                            state['learning_error']=traceback.format_exc();state['phase']='Recovery training failed'
                        continue
                    elif cmd in ('recovery_val','recovery_test'):
                        from pour_recovery_study import plan as recovery_plan,ROOT as RECOVERY_ROOT
                        if not (RECOVERY_ROOT/'best.pt').exists():
                            state['control_message']='Fit the recovery policy first';continue
                        study=recovery_plan((selected_result() or best)['spec'])
                        part='validation' if cmd=='recovery_val' else 'test'
                        cases=study[part] if part=='test' else [s for s in study[part] if s.get('expert_recovery')]
                        pending.extend(dict(s,name='BC6_'+s['name'],expert_recovery=False,
                            policy_evaluation=True,checkpoint=str(RECOVERY_ROOT/'best.pt'),
                            max_seconds=32.,grasp_deadline=8.) for s in cases)
                    elif cmd=='learning_train':
                        from pour_learning import train
                        if current is not None or pending:
                            state['control_message']='Wait for collection to finish before training'
                            continue
                        try:
                            state['trained_checkpoint']=train(state)
                        except Exception:
                            state['learning_error']=traceback.format_exc()
                            state['phase']='Training failed'
                        continue
                    elif cmd in ('learning_val','learning_test'):
                        from pour_study import plan,ROOT as STUDY_ROOT
                        if not (STUDY_ROOT/'best.pt').exists():
                            state['control_message']='Fit a visual policy before autonomous evaluation'
                            continue
                        study=plan((selected_result() or best)['spec'])
                        part='validation' if cmd=='learning_val' else 'test'
                        pending.extend(dict(s,name='BC_'+s['name'],policy_evaluation=True,
                            checkpoint=str(STUDY_ROOT/'best.pt'),max_seconds=32.,grasp_deadline=8.) for s in study[part])
                    elif cmd in ('learning_calibrate','learning_collect'):
                        from pour_study import calibration,plan
                        chosen=selected_result() or best
                        if cmd=='learning_calibrate':
                            pending.extend(calibration(chosen['spec']))
                        else:
                            study=plan(chosen['spec'])
                            seen={r['spec']['name'] for r in results if r['spec'].get('study_split') in ('train','validation') and not r['spec'].get('policy_evaluation')}
                            seen.update(s['name'] for s in pending)
                            pending.extend(s for s in study['train']+study['validation'] if s['name'] not in seen)
                    elif cmd == 'robot_replay':
                        source = json.loads((ROOT/'latest-recording.json').read_text())
                        pending.append(dict(source['spec'], name='robot_recording_replay',
                                            replay_episode=source['path']))
                    elif cmd in ('robot_full','robot_empty'):
                        chosen=selected_result() or best
                        if chosen:
                            spec=bottle_spec(chosen['spec'],small=cmd=='robot_empty')
                            pending.append(dict(spec,robot=True,protocol=6,name=cmd,max_seconds=16.,target_fraction=.86,return_flow_seconds=.23,record_cameras=True,control_period=.05,opacity=1.,roughness=.22))
                    elif cmd in ('bottle_full','bottle_empty'):
                        chosen=selected_result() or best
                        if chosen: pending.append(bottle_spec(chosen['spec'],small=cmd=='bottle_empty'))
                    elif cmd=='baseline': pending.append(candidate('baseline'))
                    elif cmd=='sweep':
                        seen={r['spec']['name'] for r in results if r['spec'].get('protocol')==3}
                        pending.extend(c for c in coarse_plan() if c['name'] not in seen)
                    elif cmd=='refine' and best: pending.extend(refinement_plan((selected_result() or best)['spec']))
                    elif cmd=='validate' and best: pending.extend(validation_plan((selected_result() or best)['spec']))
                    elif cmd=='slosh': pending.extend(slosh_plan(results))
                    elif cmd=='targeted': pending.extend(targeted_plan())
                    elif cmd=='resolution': pending.extend(resolution_plan())
                    elif cmd=='render' and best: pending.extend(render_plan((selected_result() or best)['spec']))
                    elif cmd in ('hinge','hinge_control'):
                        chosen=selected_result() or best
                        if chosen: pending.append(hinge_trial(chosen['spec'],control=cmd=='hinge_control'))
                    elif cmd=='best':
                        chosen=selected_result() or best
                        if chosen: pending.append(dict(chosen['spec'],name='replay_selected' if selected_result() else 'replay_best'))
                    pending=pending[:32]
                    state['paused']=False
        except queue.Empty: pass
        state['queue']=[p['name'] for p in pending]
        if current is None and pending and not state['paused']:
            spec=pending.pop(0);state.update(trial=spec,phase='Building '+spec['name'],sim_time=0.,metrics={},progress=0.)
            state.pop('hinge_result',None)
            state.pop('bottle_result',None)
            current=setup(spec)
        if current and not state['paused']:
            trial=current;spec=trial['spec']
            transition_start = trial['t']
            if (spec.get('expert_recovery') and trial.get('expert') is None and
                    not trial.get('recovery_error') and trial['t']>=spec['takeover_seconds']):
                try:
                    from pour_recovery import RecoveryExpert
                    trial['expert']=RecoveryExpert(trial['bottle'],trial['t'])
                    json_write(trial['folder']/'takeover.json',trial['expert'].evidence)
                except Exception:
                    trial['recovery_error']=traceback.format_exc()
                    trial['evaluation_termination']='expert_takeover_failed'
            transition_action = None
            transition_teacher_action = None
            for substep in range(round(.05/spec['dt'])):
                if trial['receiver_orient'] is not None:
                    angle,_,_=hinge_motion(trial['t']+spec['dt'],spec)
                    trial['receiver_orient'].Set(Gf.Quatf(math.cos(angle/2),Gf.Vec3f(0.,-math.sin(angle/2),0.)))
                if trial['bottle'] is not None:
                    if trial.get('expert'):
                        if substep==0:trial['expert'].drive(trial['t'])
                    elif trial.get('actor'):
                        if substep==0:
                            bot=trial['bottle']
                            bot.command=trial['actor'].act(trial['recording'].last,bot,trial['t'])
                            for drive,target in zip(bot.drives,bot.command): drive.Set(float(target))
                    elif trial['replay'] is not None:
                        if substep == 0:
                            bot = trial['bottle']
                            bot.command = trial['replay']['actions'][trial['frame']].copy()
                            bot.command_angle = float(trial['replay']['angles'][trial['frame']])
                            bot.phase = 'Replaying recorded commands'
                            for drive, target in zip(bot.drives, bot.command):
                                drive.Set(float(target))
                    else:
                        trial['bottle'].drive(trial['t']+spec['dt'])
                    if substep == 0 and trial.get('recording'):
                        transition_action = trial['bottle'].command.copy()
                        if spec.get('noisy_expert') and not spec.get('policy_evaluation'):
                            transition_teacher_action=trial['bottle'].teacher_command.copy()
                else:
                    emit(trial)
                if spec.get('slosh_speed',0)>0 and trial['t']>=spec['slosh_time'] and not trial['kicked']:
                    kp,kv=read_particles(trial)
                    pool=(abs(kp[:,0])<HALF_X)&(abs(kp[:,1])<HALF_Y)&(kp[:,2]>=0)&(kp[:,2]<WALL_HEIGHT)
                    kv[pool,0]+=spec['slosh_speed']
                    trial['inst'].GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(kv))
                    trial['kicked']=True
                    json_write(trial['folder']/'impulse.json',dict(t=trial['t'],vx=spec['slosh_speed'],count=int(pool.sum()),kind='synthetic uniform pool velocity impulse'))
                sim.step(render=False)
                trial['t']+=spec['dt']
            if trial.get('recording'):
                trial['recording'].update_camera_poses()
            sim.render()
            omni.physx.get_physx_interface().update_transformations(False,True,True,False)
            trial['frame']+=1
            p,v=read_particles(trial)
            meshprim=stage.GetPrimAtPath('/World/FluidSystem/Isosurface')
            mesh=UsdGeom.Mesh(meshprim).GetPointsAttr().Get() if meshprim else None
            measured_p,measured_v=p,v
            phase='Flowing' if trial['t']<=spec['flow_seconds'] else ('Sloshing' if trial['kicked'] else 'Settling')
            if spec.get('hinge'):
                angle,rate,phase=hinge_motion(trial['t'],spec)
                measured_p,measured_v=to_container(p,v,angle,rate)
                if mesh is not None and len(mesh): mesh=(np.asarray(mesh)-PIVOT)@hinge_rotation(angle)+PIVOT
            if trial['bottle'] is not None:
                row=trial['bottle'].observe(p,v,trial['t'])
                phase=trial['bottle'].phase
            else:
                row=sample(measured_p,measured_v,spec,trial['t'],trial['expected_count'],trial['expected_count']*spec['spacing']**3*1e6,trial['speed'],mesh)
            if spec.get('hinge'):
                row.update(hinge_angle_deg=math.degrees(angle),hinge_angular_rate_rad_s=rate,
                           edge_lift_mm=110*math.sin(angle),measurement_frame='container',
                           **obstacle_metrics(measured_p,spec['spacing']))
                row.update(floor_diagnostics(measured_p,trial.get('previous_local'),spec['spacing']))
                trial['previous_local']=measured_p.copy()
            if trial.get('recording'):
                trial['recording'].append(transition_start, transition_action, transition_start, row,
                                          teacher_action=transition_teacher_action)
                state['recording']['frames'] = trial['recording'].count
            trial['samples'].append(row)
            if spec.get('save_particle_trajectory',True) and trial['frame']%5==0: trial['trajectory'].append((trial['t'],p,v))
            state.update(phase=phase,sim_time=round(trial['t'],2),metrics=row,
                         progress=min(1.,trial['t']/(spec['flow_seconds']+spec['settle_seconds'])))
            preview.capture()
            if preview.jpeg and trial['frame']%2==0:
                (trial['folder']/'frames'/f'{trial["frame"]:04}.jpg').write_bytes(preview.jpeg)
            if trial['bottle'] is not None:
                state['progress']=min(1.,trial['t']/spec['max_seconds'])
                complete=((trial['actor'].complete() or trial['t']>=spec['max_seconds']) if trial.get('actor') else
                          trial['frame'] >= len(trial['replay']['actions']) if trial['replay'] is not None
                          else trial['bottle'].complete(trial['t']))
                if spec.get('expert_recovery'):
                    complete=bool(trial.get('recovery_error') or trial['t']>=spec['max_seconds'] or
                                  (trial.get('expert') and trial['expert'].complete(trial['t'])))
                if (trial.get('actor') and trial['t']>=spec.get('grasp_deadline',8.) and
                        not trial['bottle'].verified):
                    complete=True
                    trial['evaluation_termination']='grasp_deadline_exceeded'
            else:
                complete=trial['t']>=spec['flow_seconds']+spec['settle_seconds']-1e-6
            if complete:
                state['progress']=1.

                finish(trial);current=None
                state['phase']=('Complete' if results[-1]['metrics']['eligible'] else 'Trial finished with failed checks') if spec.get('robot') else 'Complete; pool held for inspection'
        else:
            if sim: sim.render()
            else: app.update()
            preview.capture()
            time.sleep(.03)
except Exception:
    if current and current.get('recording'):
        current['recording'].abort(traceback.format_exc())
    state['phase']='Error';state['error']=traceback.format_exc()
    json_write(OUT/'error.json',state)
    print(state['error'],flush=True)
    raise
finally:
    app.close()
