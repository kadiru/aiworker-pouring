"""Narrow imports in the disposable recovered checkout; never edit the backup."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent / "robotis_lab"
changes = []
# AppleDouble metadata is not Python/Kit source; retain it only in the archive.
for metadata in root.rglob("._*"):
    if metadata.is_file(): metadata.unlink()
for relative in (
    "source/robotis_lab/robotis_lab/__init__.py",
    "source/robotis_lab/robotis_lab/simulation_tasks/__init__.py",
    "source/robotis_lab/robotis_lab/simulation_tasks/manager_based/FFW_BG2/pick_place/__init__.py",
    "third_party/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/__init__.py",
):
    path = root / relative
    backup = path.with_name(path.name + ".recovered")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    source = backup.read_text()
    if relative.endswith("pick_place/__init__.py"):
        source = '"""Only shared pick/place MDP helpers are needed by this pouring profile."""\n'
    elif relative.endswith("robotis_lab/__init__.py"):
        source = source.replace("from .simulation_tasks import *", "# Tasks are imported explicitly by run_quest.py.")
        source = source.replace("from .real_world_tasks import *", "# Physical robot / DDS integration is not used in this simulation.")
        source = source.replace("from .ui_extension_example import *", "# The isolated runtime owns its browser controls.")
    else:
        source = source.replace("import_packages(__name__, _BLACKLIST_PKGS)", "# Isolated restoration: import only the requested task.")
    path.write_text(source)
    changes.append(relative)
path = root / "source/robotis_lab/robotis_lab/devices/quest3_controller.py"
backup = path.with_name(path.name + ".recovered")
if not backup.exists(): backup.write_bytes(path.read_bytes())
source = backup.read_text()
source = source.replace(
    'if ctrl is None or getattr(ctrl, "size", 0) == 0:',
    'if (ctrl is None or getattr(ctrl, "shape", None) != (2, 7)\n'
    '                or not np.isfinite(ctrl).all() or np.linalg.norm(ctrl[0, 3:7]) < 1e-6):')
path.write_text(source)
changes.append(str(path.relative_to(root)))
print(json.dumps({"modified_runtime_copies": changes}))
