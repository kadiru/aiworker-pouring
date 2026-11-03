"""Inspect the recovered USD in the isolated Isaac Sim container, without stepping physics."""
import json
from pathlib import Path
import subprocess

EXPECTED_GPU = "GPU-cde23916-c5fa-f18e-a343-127f732cc2f6"
visible = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).strip().splitlines()
if visible != [EXPECTED_GPU]:
    raise RuntimeError(f"Refusing to start: unexpected GPU visibility {visible}")

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True, "width": 960, "height": 540,
    "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
    "extra_args": ["--/plugins/carb.tasking.plugin/threadCount=4", "--/app/asyncRendering=false"],
})
from pxr import Usd, UsdGeom, UsdPhysics

path = "/workspace/assets/FFW_BG2_without_mimic.usd"
stage = Usd.Stage.Open(path)
report = {
    "gpu": visible, "asset": path, "default_prim": str(stage.GetDefaultPrim().GetPath()),
    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
    "roots": [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()],
    "articulations": [], "fixed_joints": [], "joints": [], "layers": [],
}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        report["articulations"].append(str(prim.GetPath()))
    if prim.IsA(UsdPhysics.Joint):
        joint = UsdPhysics.Joint(prim)
        item = {"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()]}
        report["joints"].append(item)
        if prim.IsA(UsdPhysics.FixedJoint):
            report["fixed_joints"].append(item)
report["layers"] = [p.identifier for p in stage.GetUsedLayers()]
bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
box = bounds.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
report["bounds"] = {"min": list(box.GetMin()), "max": list(box.GetMax())}
Path("/workspace/outputs/robot-inspection.json").write_text(json.dumps(report, indent=2) + "\n")
print("AIWORKER_ASSET_INSPECTED=" + json.dumps(report), flush=True)
app.close()
