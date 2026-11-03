"""Recovered Quest input + ROBOTIS 7D pouring task, one isolated simulation."""
import faulthandler
import json
import os
import queue
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path
from pour_geometry import SOURCE_SCALE, SOURCE_POSITION, TARGET_POSITION, LIQUIDS

GPU = "GPU-cde23916-c5fa-f18e-a343-127f732cc2f6"
FLUID = os.environ.get("AIWORKER_FLUID", "0") == "1"
OUT = Path("/workspace/outputs")
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)
visible = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().splitlines()
if visible != [GPU]:
    raise RuntimeError(f"GPU isolation failed: {visible}")
from quest_preview import Preview
state = {"ready": False, "robot": "FFW-BG2", "profile": "quest-pouring", "gpu_uuid": GPU,
         "mode": "Quest teleoperation; dry pouring scene; no learned policy", "frame": 0,
         "recording": False, "samples": 0, "saved_episodes": 0,
         "controllers": {"left": False, "right": False}, "paused": False,
         "phase": "starting Isaac Sim 5.1 and recovered Isaac Lab 2.3"}
preview = Preview(state)

from isaaclab.app import AppLauncher
launcher = AppLauncher({
    "headless": True, "xr": True, "enable_cameras": True, "video": True,
    "device": "cpu" if FLUID else "cuda:0", "device_explicit": True, "width": 1024, "height": 576,
    "window_width": 1024, "window_height": 576, "multi_gpu": False,
    "kit_args": "--reset-user --/plugins/carb.tasking.plugin/threadCount=8 "
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                "--/renderer/multiGpu/enabled=false --/log/file=/workspace/logs51/quest-kit.log",
})
app = launcher.app
print("QUEST_APP_READY", flush=True)

import numpy as np
import torch
import omni.usd
from pxr import UsdPhysics, UsdGeom, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, EventTermCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.xr.core import XRCore
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.pour_cup.ik_rel_env_cfg import PourCupFFWBG2EnvCfg
from robotis_lab.devices.quest3_controller import Quest3ControllerRelRetargeter, CONTROLLER_RIGHT, CONTROLLER_LEFT

cfg = PourCupFFWBG2EnvCfg()
cfg.scene.num_envs = 1
cfg.seed = 42
cfg.sim.device = "cpu" if FLUID else "cuda:0"
cfg.sim.dt = 0.005 if FLUID else 0.01
cfg.sim.render_interval = 10 if FLUID else 5
cfg.decimation = cfg.sim.render_interval
cfg.viewer.eye = (3.1, -3.1, 2.1)
cfg.viewer.lookat = (0.3, 0.0, 0.85)
cfg.sim.render.antialiasing_mode = "DLSS"
cfg.sim.render.enable_translucency = True
cfg.sim.render.enable_reflections = True
cfg.events.spawn_fluid_cup = None  # Missing historical extension; dry cups are explicit below.
cfg.events.reset_objects = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
cfg.scene.robot.spawn.usd_path = "/workspace/assets/FFW_BG2_without_mimic.usd"
# Keep original contact geometry for manipulation, rather than the preview's hull simplification.
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
cfg.scene.left_wrist_cam = None
cfg.observations.policy.left_wrist_cam = None
for camera_name in ("right_wrist_cam", "head_cam", "overhead_cam"):
    getattr(cfg.scene, camera_name).update_period = 0.1
cfg.scene.head_cam.data_types = ["rgb"]
cfg.scene.overhead_cam.offset.pos = (0.70, 0.0, 1.95)
cfg.scene.overhead_cam.offset.rot = (0.0, 1.0, 0.0, 0.0)
cfg.scene.overhead_cam.offset.convention = "ros"
cfg.scene.overhead_cam.spawn.clipping_range = (0.05, 4.0)
cfg.scene.table = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0, 0.96)),
    spawn=sim_utils.CuboidCfg(size=(0.85, 1.05, 0.08), collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.003,rest_offset=0.),
                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.29, 0.33))))
cfg.scene.plane = AssetBaseCfg(
    prim_path="/World/Ground", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.03)),
    spawn=sim_utils.CuboidCfg(size=(10, 10, 0.06), collision_props=sim_utils.CollisionPropertiesCfg(),
                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.23))))
for index, (x, y) in enumerate(((0.385,-0.465),(0.385,0.465),(1.115,-0.465),(1.115,0.465))):
    setattr(cfg.scene, f"table_leg{index}", AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/TableLeg{index}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x,y,0.47)),
        spawn=sim_utils.CuboidCfg(size=(0.045,0.045,0.94),
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18,0.20,0.23)))))
for spec in LIQUIDS:
    name, position, scale = spec['cup'], spec['position'], SOURCE_SCALE
    setattr(cfg.scene, name, RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.UsdFileCfg(usd_path="/workspace/assets/source_cup_open.usda", scale=scale,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec['rgb'], opacity=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0))))
cfg.terminations.time_out = None
cfg.terminations.success = None
cfg.observations.policy.concatenate_terms = False
cfg.env_name = "RobotisLab-PourCup-FFW-BG2-IK-Rel-v0"
cfg.recorders = ActionStateRecorderManagerCfg()
cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
cfg.recorders.export_in_record_pre_reset = False
cfg.recorders.dataset_export_dir_path = "/workspace/datasets"
cfg.recorders.dataset_filename = "quest_pouring_" + time.strftime("%Y%m%d_%H%M%S")
if FLUID:
    import pour_fluid
    cfg.sim.use_fabric = False
    cfg.events.spawn_fluid_cup = EventTermCfg(func=pour_fluid.spawn_fluid, mode="prestartup")
    cfg.events.reset_fluid = EventTermCfg(func=pour_fluid.reset_fluid, mode="reset")
    app.set_setting("/physics/updateParticlesToUsd", True)
    app.set_setting("/physics/updateToUsd", True)
    app.set_setting("/physics/updateVelocitiesToUsd", True)
    state["mode"] = "Quest teleoperation; PhysX PBD liquid with smooth water surface; no learned policy"
state["cup_geometry"] = {"source_scale": SOURCE_SCALE, "source_wall_diameter_m": .104 * SOURCE_SCALE[0],
                         "source_height_m": .16 * SOURCE_SCALE[2], "target_wall_diameter_m": .104*SOURCE_SCALE[0], "cup_count":5}
if FLUID: state['liquid_comparison'] = pour_fluid.metadata()
state["fluid_enabled"] = FLUID
state["recorded_cameras"] = ["right_wrist_cam", "head_cam", "overhead_cam"]
state["dataset"] = cfg.recorders.dataset_filename + ".hdf5"
state["phase"] = "loading robot contact geometry and one pouring environment"

# Disable the recovered asset's duplicate world joint in its new scene layer.
original_spawn = cfg.scene.robot.spawn.func
def spawn_robot(path, spawn_cfg, **kwargs):
    result = original_spawn(path, spawn_cfg, **kwargs)
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if str(prim.GetPath()).endswith("ffw_bg2_follower/world/FixedJoint"):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    return result
cfg.scene.robot.spawn.func = spawn_robot
env = ManagerBasedRLEnv(cfg=cfg)
if FLUID:
    env.sim.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True, update_velocities_to_usd=True)
    app.set_setting("/physics/updateParticlesToUsd", True)
    app.set_setting("/physics/suppressReadback", False)
    # Isaac Core's direct simulate/fetch path skips fluid post-processing in
    # this runtime. Use the full PhysX step, once per existing physics step.
    pour_fluid.install_surface_cadence(env)
robot = env.scene["robot"]
state["num_dof"] = robot.num_joints
state["dof_names"] = robot.joint_names
state["collision_profile"] = "original convex decompositions and primitive open cups"
state["phase"] = "creating Quest controller input"
device_cfg = cfg.teleop_devices.devices["quest3controller"]
device_cfg.sim_device = env.device
for retarget in device_cfg.retargeters:
    retarget.sim_device = env.device
device = create_teleop_device("quest3controller", cfg.teleop_devices.devices,
    {"START": lambda: preview.commands.put_nowait("start"),
     "STOP": lambda: preview.commands.put_nowait("save"),
     "RESET": lambda: preview.commands.put_nowait("discard")})

# Observe the actual controller data used by the recovered retargeter.
original_raw = device._get_raw_data
def observed_raw():
    raw = original_raw()
    for side, key in (("left", CONTROLLER_LEFT), ("right", CONTROLLER_RIGHT)):
        ctrl = raw.get(key)
        tracked = ctrl is not None and getattr(ctrl, "size", 0) > 0
        state["controllers"][side] = bool(tracked)
        if tracked:
            state[side + "_controller"] = ctrl.tolist()
            state["last_controller_unix"] = time.time()
    return raw
device._get_raw_data = observed_raw
probe_retarget = Quest3ControllerRelRetargeter(device_cfg.retargeters[0])

running = True
probe = None
reset_count = 0
last_write = 0
event_log = Path("/workspace/datasets") / (cfg.recorders.dataset_filename + ".events.jsonl")
def event(name, **details):
    with event_log.open("a") as stream:
        stream.write(json.dumps({"event": name, "unix": time.time(), "frame": state["frame"], **details}) + "\n")

def reset_scene():
    global reset_count, probe
    env.recorder_manager.reset()
    env.reset()
    env.recorder_manager.reset()
    device.reset()
    for retarget in device._retargeters:
        if hasattr(retarget, "reset_reference"): retarget.reset_reference()
    probe_retarget.reset_reference()
    state.update(recording=False, samples=0)
    state.pop('spill_test', None)
    reset_count += 1
    state["reset_count"] = reset_count
    event("reset", reset_count=reset_count)

def start_recording():
    if state["recording"]: return
    env.recorder_manager.reset()
    env.recorder_manager.record_post_reset([0])
    state.update(recording=True, samples=0,
                 recording_input_source='synthetic_validation' if probe or state.get('spill_test') else 'quest_or_browser')
    event("start", input_source=state['recording_input_source'])

def save_recording():
    if state["recording"] and state["samples"]:
        # Saving does not assert a measured successful pour. Export all attempts.
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[False]], device=env.device))
        env.recorder_manager.export_episodes([0])
        handler = env.recorder_manager._dataset_file_handler
        saved = handler._hdf5_data_group[f"demo_{handler._demo_count-1}"]
        saved.attrs["input_source"] = state['recording_input_source']
        saved.attrs["evidence_kind"] = "simulation"
        saved.attrs["outcome_verified"] = False
        saved.attrs["success_label"] = "not_evaluated"
        if FLUID:
            saved.attrs['fluid_conditions_json'] = json.dumps(pour_fluid.metadata())
            saved.attrs['fluid_surface_profile'] = env._pour_surface_profile
        handler.flush()
        shutil.copyfile(Path("/workspace/datasets") / state["dataset"], OUT / "quest-last-export.hdf5")
        state["saved_episodes"] += 1
        event("operator_saved", samples=state["samples"], outcome_verified=False,
              input_source=state['recording_input_source'])
    reset_scene()

def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
reset_scene()
set_camera_view(eye=np.array(cfg.viewer.eye), target=np.array(cfg.viewer.lookat))
state.update(ready=True, phase="ready; waiting for Quest controllers")
print("QUEST_SCENE_READY", flush=True)

try:
    with torch.inference_mode():
        while running and app.is_running():
            started = time.monotonic()
            action = device.advance()
            while not preview.commands.empty():
                command = preview.commands.get_nowait()
                if command == "start": start_recording()
                elif command == "save":
                    save_recording()
                    action = torch.tensor([0,0,0,0,0,0,1.0], device=env.device)
                elif command in ("discard", "reset"):
                    event("discard", samples=state["samples"])
                    probe = None
                    reset_scene()
                    action = torch.tensor([0,0,0,0,0,0,1.0], device=env.device)
                elif command == "pause":
                    state["paused"] = not state["paused"]
                    if state["paused"]: env.sim.pause()
                    else: env.sim.play()
                    for retarget in device._retargeters:
                        if hasattr(retarget, "reset_reference"): retarget.reset_reference()
                elif command == "camera": set_camera_view(eye=np.array(cfg.viewer.eye), target=np.array(cfg.viewer.lookat))
                elif command == "cup_view": set_camera_view(eye=np.array([.88,-.52,1.72]), target=np.array([.70,-.08,1.05]))
                elif command == "water_view": set_camera_view(eye=np.array([.69,-.23,1.42]), target=np.array([.68,-.22,1.04]))
                elif command == 'comparison_view': set_camera_view(eye=np.array([1.52,-.96,1.93]), target=np.array([.70,0,1.04]))
                elif command == 'spill_view': set_camera_view(eye=np.array([1.08,-.29,1.40]), target=np.array([.94,0,1.005]))
                elif command in ('surface_legacy','surface_thin') and FLUID:
                    # Kit's importer caches this standalone module. Reload only
                    # this fixed, owned source for bounded surface diagnostics.
                    source = Path(pour_fluid.__file__).read_text()
                    exec(compile(source,pour_fluid.__file__,'exec'),pour_fluid.__dict__)
                    pour_fluid.surface_profile(env, legacy=command=='surface_legacy')
                elif command.startswith('spill_') and command[-1:].isdigit() and FLUID and not state['recording']:
                    reset_scene()
                    index = int(command[-1])-1
                    state['spill_test'] = dict(id=index+1,start_sim_time=float(env.sim.current_time),
                        phase='settling', release_at_sim_time=float(env.sim.current_time)+1.0,
                        evidence_kind='synthetic standardized release, not a robot pour')
                    event('synthetic_spill',liquid=index+1)
                elif command == "start_xr":
                    XRCore.request_enable_profile("ar")
                elif command in ("probe", "motion") and not state["recording"]:
                    reset_scene()
                    probe = {"step": 0, "initial_q": robot.data.joint_pos[0].cpu().tolist(), "samples": []}
                    start_recording()
            if not state["paused"]:
                if state.get('spill_test'):
                    action = torch.tensor([0,0,0,0,0,0,1.0], device=env.device)
                if probe is not None:
                    n = probe["step"]
                    # Deterministic synthetic controller poses, through the saved retargeter and IK.
                    offset = max(0, min(n-20, 40)) * 0.0004 - max(0, min(n-80, 40)) * 0.0004
                    angle = max(0, min(n-20, 40)) * 0.004 - max(0, min(n-80, 40)) * 0.004
                    ctrl = np.array([[0.4, -0.2, 1.2+offset, np.cos(angle/2), np.sin(angle/2), 0, 0],
                                     [0, 0, 1, 1 if 40 <= n < 70 else 0, 0, 0, 0]], dtype=np.float32)
                    action = probe_retarget.retarget({CONTROLLER_RIGHT: ctrl})
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite teleoperation action")
                env.step(action.reshape(1, -1))
                if FLUID:
                    # Isaac Lab steps the PhysX simulation interface directly;
                    # explicitly publish particle positions before diagnostics/recording.
                    import omni.physx
                    omni.physx.get_physx_interface().update_transformations(False, True, True, False)
                state["frame"] += 1
                test = state.get('spill_test')
                if test and test['phase']=='settling' and env.sim.current_time>=test['release_at_sim_time']:
                    pour_fluid.spill(env,test['id']-1)
                    test.update(phase='released',start_sim_time=float(env.sim.current_time))
                if state["recording"]:
                    state["samples"] += 1
                    env.recorder_manager.add_to_episodes("timestamps_unix", torch.tensor([[time.time()]],dtype=torch.float64,device=env.device))
                    if FLUID:
                        env.recorder_manager.add_to_episodes("fluid_positions", torch.tensor(pour_fluid.positions(env)[None], device=env.device))
                    if state["samples"] >= 1200 and probe is None:
                        event("duration_limit", limit_samples=1200)
                        save_recording()
                else:
                    env.recorder_manager.reset()
                if probe is not None:
                    probe["samples"].append(robot.data.joint_pos[0].cpu().tolist())
                    probe["step"] += 1
                    if probe["step"] >= 160:
                        probe["final_q"] = robot.data.joint_pos[0].cpu().tolist()
                        probe["evidence_kind"] = "simulation; synthetic controller input, not Quest hardware"
                        (OUT / "quest-ik-probe.json").write_text(json.dumps(probe, indent=2))
                        save_recording()
                        probe = None
                        state["probe_completed"] = True
            else:
                app.update()
            state["updated_unix"] = time.time()
            xr_core = XRCore.get_singleton()
            state["xr_enabled"] = bool(xr_core and xr_core.is_xr_enabled())
            state["xr_display_enabled"] = bool(xr_core and xr_core.is_xr_display_enabled())
            state["sim_time_s"] = float(env.sim.current_time)
            state["joint_positions"] = robot.data.joint_pos[0].cpu().tolist()
            state["source_cup_pose"] = env.scene["source_cup"].data.root_state_w[0, :7].cpu().tolist()
            state["target_cup_pose"] = env.scene["target_cup"].data.root_state_w[0, :7].cpu().tolist()
            if FLUID: state["fluid"] = pour_fluid.diagnostics(env)
            if not np.isfinite(state["joint_positions"]).all():
                raise RuntimeError("Non-finite robot state")
            preview.capture()
            if time.monotonic() - last_write > 2:
                import io
                from PIL import Image
                camera_stats = {}
                for camera_name in state["recorded_cameras"]:
                    pixels = env.scene[camera_name].data.output["rgb"][0].cpu().numpy()[:,:,:3]
                    encoded = io.BytesIO()
                    Image.fromarray(pixels).save(encoded, "JPEG", quality=85)
                    preview.cameras[camera_name] = encoded.getvalue()
                    (OUT / f"quest-{camera_name}.jpg").write_bytes(encoded.getvalue())
                    camera_stats[camera_name] = {"shape":list(pixels.shape),"mean":float(pixels.mean()),"std":float(pixels.std())}
                state["camera_stats"] = camera_stats
                if FLUID: state["fluid_render"] = pour_fluid.render_diagnostics(env)
                (OUT / "quest-status.json").write_text(json.dumps(state, indent=2))
                last_write = time.monotonic()
            time.sleep(max(0, 0.05 - (time.monotonic()-started)))
except Exception as error:
    traceback.print_exc()
    state.update(ready=False, error=repr(error))
    (OUT / "quest-error.json").write_text(json.dumps(state, indent=2))
    raise
finally:
    env.close()
    preview.server.shutdown()
    app.close()
