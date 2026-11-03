"""Live, single-robot dry scene using the recovered BG2 asset. No policy or liquid."""
import json
import ctypes
import io
import os
import traceback
import faulthandler
import signal
import subprocess
import time
import queue
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

GPU = "GPU-cde23916-c5fa-f18e-a343-127f732cc2f6"
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
visible = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().splitlines()
if visible != [GPU]:
    raise RuntimeError(f"GPU isolation failed: {visible}")

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True, "hide_ui": False, "width": 1280, "height": 720,
    "window_width": 1280, "window_height": 720,
    "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
    "renderer": "RaytracedLighting",
    "anti_aliasing": 1, "limit_cpu_threads": 8,
    "extra_args": ["--reset-user"],
})
from isaacsim.core.experimental.utils.app import enable_extension
enable_extension("isaacsim.core.api")
print("AIWORKER_CORE_ENABLED", flush=True)
app.set_setting("/app/window/drawMouse", True)
app.set_setting("/exts/omni.kit.livestream.app/primaryStream/publicIp", "192.168.68.60")
app.set_setting("/exts/omni.kit.livestream.app/primaryStream/signalPort", 49110)
app.set_setting("/exts/omni.kit.livestream.app/primaryStream/streamPort", 48008)
app.set_setting("/exts/omni.kit.livestream.app/primaryStream/targetFps", 30)
if os.environ.get("AIWORKER_WEBRTC") == "1":
    enable_extension("omni.kit.livestream.app")
    print("AIWORKER_STREAM_ENABLED", flush=True)

import numpy as np
import omni.ui as ui
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.types import ArticulationAction
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_buffer
from PIL import Image

OUT = Path("/workspace/outputs")
OUT.mkdir(exist_ok=True)
world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/30, backend="numpy", device="cpu")
print("AIWORKER_WORLD_CREATED", flush=True)
stage = omni.usd.get_context().get_stage()
robot_prim = add_reference_to_stage("/workspace/assets/FFW_BG2_without_mimic.usd", "/World/Robot")
print("AIWORKER_USD_REFERENCED", flush=True)
articulation_path = "/World/Robot/ffw_bg2_follower/root_joint"
root_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(articulation_path))
if not root_joint or root_joint.GetBody0Rel().GetTargets():
    raise RuntimeError("Expected recovered fixed base attached to world")
redundant = stage.GetPrimAtPath("/World/Robot/ffw_bg2_follower/world/FixedJoint")
if redundant:
    UsdPhysics.Joint(redundant).CreateJointEnabledAttr(False)
for prim in Usd.PrimRange(robot_prim):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    # The recovered asset decomposes dense visual meshes (up to 280k vertices).
    # Use fast hulls for this dry startup profile; preserve the original USD.
    # This profile is not suitable for validating grasp/contact fidelity.
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        collision = UsdPhysics.MeshCollisionAPI(prim)
        if collision.GetApproximationAttr().Get() == "convexDecomposition":
            collision.CreateApproximationAttr("convexHull")
articulation = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(articulation_path))
articulation.CreateSolverPositionIterationCountAttr(32)
articulation.CreateSolverVelocityIterationCountAttr(1)
robot = world.scene.add(Robot(prim_path=articulation_path, name="aiworker"))
world.scene.add(GroundPlane(prim_path="/World/Ground", name="ground", size=10, color=np.array([0.18,0.20,0.23])))
# A simple new tabletop fixture, not a recovered calibrated manipulation scene.
world.scene.add(FixedCuboid(prim_path="/World/Table", name="table", position=np.array([0.75,0,0.96]),
                           scale=np.array([0.85,1.05,0.08]), color=np.array([0.25,0.29,0.33])))
for index, (x, y) in enumerate(((0.385,-0.465),(0.385,0.465),(1.115,-0.465),(1.115,0.465))):
    world.scene.add(FixedCuboid(prim_path=f"/World/TableLeg{index}", name=f"table_leg_{index}",
                               position=np.array([x,y,0.47]), scale=np.array([0.045,0.045,0.94]),
                               color=np.array([0.18,0.20,0.23])))
for index, (filename, xy) in enumerate((("source_cup_open.usda", (0.68,-0.22)), ("target_cup_open.usda", (0.73,0.14)))):
    path = f"/World/Cup{index}"
    prim = add_reference_to_stage(f"/workspace/assets/{filename}", path)
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(prim).ComputeAlignedRange()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], 1.005 - box.GetMin()[2]))
light = UsdLux.DomeLight.Define(stage, "/World/Light")
light.CreateIntensityAttr(1800)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(1600)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, -20, 25))
set_camera_view(eye=np.array([3.1,-3.1,2.1]), target=np.array([0.3,0,0.85]))

print("AIWORKER_SCENE_CREATED", flush=True)
world.reset()
print("AIWORKER_PHYSICS_RESET", flush=True)
names = list(robot.dof_names)
home = np.zeros(robot.num_dof, dtype=np.float32)  # Zero pose from saved FFW_BG2_CFG.
robot.set_joint_positions(home)
robot.set_joint_velocities(np.zeros_like(home))
kp, kd = [], []
for name in names:
    if name == "lift_joint": gains = (10000,100)
    elif name.startswith("head_"): gains = (150,3)
    elif name.startswith("gripper_"): gains = (100,4) if name.endswith("joint1") else (2,0.5)
    elif name.endswith("joint7"): gains = (200,3)
    else: gains = (600,30) if name.endswith(("joint1","joint2")) else (600,20)
    kp.append(gains[0]); kd.append(gains[1])
robot.get_articulation_controller().set_gains(kps=np.array(kp), kds=np.array(kd))
robot.apply_action(ArticulationAction(joint_positions=home.copy()))
stage.GetRootLayer().Export(str(OUT / "aiworker-dry-scene.usda"))

state = {"ready": False, "robot": "FFW-BG2", "gpu_uuid": GPU, "physics": "CPU PhysX",
         "collision_profile": "convex hulls for dry startup; original asset preserved",
         "viewer": "live image preview, up to 5 fps; native WebRTC disabled",
         "fixed_base": True, "num_dof": robot.num_dof, "dof_names": names,
         "mode": "scripted dry startup check; no policy or liquid", "frame": 0}
running = True
paused = False
test_start = 0
head_index = names.index("head_joint1")
motion_samples = []
motion_saved = False
commands = queue.SimpleQueue()
preview_bytes = b""
capture_pending = False
capture_handle = None
capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
capsule_pointer.restype = ctypes.c_void_p
capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

def on_preview(buffer, buffer_size, width, height, byte_format):
    global preview_bytes, capture_pending
    try:
        if buffer_size != width * height * 4:
            raise ValueError(f"Unexpected viewport pixel layout: {byte_format}, {buffer_size}")
        raw = ctypes.string_at(capsule_pointer(buffer, None), buffer_size)
        frame_image = Image.frombytes("RGBA", (width, height), raw).convert("RGB")
        encoded = io.BytesIO()
        frame_image.save(encoded, format="JPEG", quality=85)
        preview_bytes = encoded.getvalue()
        state["preview_updated_unix"] = time.time()
        state["preview_frame"] = state["frame"]
        state.pop("preview_error", None)
    except Exception as error:
        state["preview_error"] = repr(error)
    finally:
        capture_pending = False

def stop(*_):
    global running
    running = False

def reset_pose():
    global test_start, motion_saved, motion_samples
    test_start = state["frame"] - 1000
    motion_saved = True
    motion_samples = []
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros_like(home))
    robot.apply_action(ArticulationAction(joint_positions=home.copy()))

def toggle_pause():
    global paused
    paused = not paused
    if paused: world.pause()
    else: world.play()

def motion_check():
    global test_start, motion_samples, motion_saved
    test_start = state["frame"]
    motion_samples = []
    motion_saved = False

window = ui.Window("AIWorker — RTX 4070", width=310, height=180)
with window.frame:
    with ui.VStack(spacing=6):
        ui.Label("FFW-BG2 | recovered robot | dry scene")
        ui.Label("One robot. No liquid or learned policy.")
        ui.Button("Small head-motion check", clicked_fn=motion_check)
        ui.Button("Reset robot pose", clicked_fn=reset_pose)
        ui.Button("Pause / resume physics", clicked_fn=toggle_pause)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/workspace/viewer", **kwargs)
    def log_message(self, *_):
        pass
    def do_POST(self):
        if self.path != "/control" or self.headers.get("Origin") not in (None, "http://192.168.68.60:8212"):
            self.send_error(403); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024: raise ValueError("Invalid request size")
            command = json.loads(self.rfile.read(length))["command"]
            if command not in {"motion", "reset", "pause", "camera"}: raise ValueError("Unknown control")
            commands.put(command)
        except (ValueError, KeyError):
            self.send_error(400); return
        self.send_response(202); self.end_headers()
    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            body = json.dumps(state).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        elif self.path.split("?")[0] == "/snapshot.jpg":
            body = preview_bytes
            if not body: self.send_error(503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

server = ThreadingHTTPServer(("192.168.68.60", 8212), Handler)
Thread(target=server.serve_forever, daemon=True).start()
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
print("AIWORKER_READY=" + json.dumps(state), flush=True)
last_report = time.monotonic()
last_capture = 0.0

try:
    while running and app.is_running():
        started = time.monotonic()
        while not commands.empty():
            command = commands.get()
            if command == "motion": motion_check()
            elif command == "reset": reset_pose()
            elif command == "pause": toggle_pause()
            elif command == "camera": set_camera_view(eye=np.array([3.1,-3.1,2.1]), target=np.array([0.3,0,0.85]))
        frame = state["frame"]
        elapsed = frame - test_start
        if not paused:
            target = home.copy()
            if 60 <= elapsed < 120: target[head_index] = 0.08
            robot.apply_action(ArticulationAction(joint_positions=target))
            # World advances two 1/60 s physics ticks per 1/30 s rendered step.
            world.step(render=True)
            if frame == 0: print("AIWORKER_FIRST_RENDER_STEP", flush=True)
            if frame == 10:
                state["ready"] = True
            if frame == 20: faulthandler.cancel_dump_traceback_later()
            q = robot.get_joint_positions()
            if not np.isfinite(q).all(): raise RuntimeError("Non-finite robot joint state")
            if elapsed <= 180:
                motion_samples.append({"frame": frame, "target_rad": float(target[head_index]), "measured_rad": float(q[head_index])})
            if elapsed == 180 and not motion_saved:
                peak = max(abs(s["measured_rad"]) for s in motion_samples)
                result = {"joint": "head_joint1", "target_rad": 0.08, "peak_rad": peak,
                          "return_error_rad": abs(float(q[head_index])),
                          "passed": peak > 0.05 and abs(float(q[head_index])) < 0.02, "samples": motion_samples}
                (OUT / "joint-motion-check.json").write_text(json.dumps(result, indent=2)+"\n")
                state["motion_check_passed"] = result["passed"]
                motion_saved = True
                print("AIWORKER_MOTION_CHECK=" + json.dumps({k:v for k,v in result.items() if k != "samples"}), flush=True)
        else:
            app.update()
        if state["ready"] and time.monotonic() - last_capture >= 0.2 and not capture_pending:
            capture_pending = True
            capture_handle = capture_viewport_to_buffer(get_active_viewport(), on_preview)
            last_capture = time.monotonic()
        if not paused: state["frame"] += 1
        state["sim_time_s"] = float(world.current_time)
        state["paused"] = paused
        state["updated_unix"] = time.time()
        if time.monotonic() - last_report > 2:
            (OUT / "status.json").write_text(json.dumps(state, indent=2)+"\n")
            last_report = time.monotonic()
        time.sleep(max(0, 1/30 - (time.monotonic()-started)))
except Exception as error:
    traceback.print_exc()
    (OUT / "runtime-error.json").write_text(json.dumps({"error": repr(error), "time": time.time()}))
    raise
finally:
    server.shutdown()
    app.close()
