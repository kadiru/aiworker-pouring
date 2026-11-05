"""Read-only enumeration of geometry actually covered by the preflight."""
import json
from check_workspace_clearance import geometry
from pxr import Usd, UsdPhysics, UsdGeom

stage,planner,meshes=geometry()
checked={m[1] for m in meshes}
all_robot={str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)}
print(json.dumps(dict(checked_meshes=sorted(checked),omitted_robot_collision_meshes=sorted(all_robot-checked)),indent=2))
