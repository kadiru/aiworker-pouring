# Recovered AIWorker setup

Recovered October 28, 2026, America/Los_Angeles. SSH succeeded after the user installed the Mac mini's public key. The recovery phase was read-only. The user subsequently authorized a simulator restart; the isolated RTX 4070 baseline is now running with verified motion and a live preview. See [runtime details and limits](../runtime/README.md).

Later the same day, the Quest/pouring server was restored with the recovered Isaac Lab/ROBOTIS source, Sim 5.1, CloudXR, three cameras, recording and actual PBD liquid. See [the current Quest runbook](QUEST_RESTORE.md). Simulator checks passed; actual headset connection and a teleoperated pour remain unverified. The historical inventories and limitations below describe the recovery stage, before that restoration.

## Original project and robot

The original project is **`/Users/kadirumacbook/Documents/aiworker`** on **`kadirumacbook@192.168.68.59`**. It contains the June–July work, a June 22 handoff, Quest 3 runbook, fluid scripts, pouring patches, and Dell/ROG backups. Git HEAD is `c5a74c68acb815d3263253622f55294616ef0e07`; modified and untracked files were preserved. The MacBook held code, controls and backups; simulation ran on Linux/NVIDIA hosts.

The saved robot is **FFW-BG2**, selected through `FFW_BG2_WITHOUT_MIMIC_CFG` and **`data/robots/FFW/FFW_BG2_without_mimic.usd`**. This replaces the earlier tentative identification. Three saved copies of that USD are identical: 38,359,462 bytes, SHA-256 `82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d`. The binary USD was subsequently copied into `runtime/assets/` with the same hash. Initial metadata is saved in [the asset inspection](../recovered/macbook-2026-10-28/robot-asset-inspection.json); [runtime stage inspection](../evidence/runtime-robot-inspection.json) confirms the fixed world root and 25 movable joints. The running scene disables the extra redundant fixed joint in its own layer; the recovered asset remains unchanged.

## Control and task inventory

| Component | Recovered evidence |
| --- | --- |
| Single-arm baseline | `RobotisLab-PickPlace-FFW-BG2-IK-Rel-v0`, `quest3controller` |
| Bimanual baseline | `RobotisLab-PickPlace-FFW-BG2-Bimanual-IK-Rel-v0`, `quest3bimanual` |
| Pouring registration | `RobotisLab-PourCup-FFW-BG2-IK-Rel-v0` |
| Pouring action | Right-arm relative pose IK, damped least squares, seven arm joints, binary four-joint gripper |
| Quest workflow | June 24 runbook: browser-based Isaac Teleop 1.0.193 / CloudXR Runtime 6.1.0, A/B/X recording workflow |
| Baseline cameras | Left/right wrist RGB and head RGB/depth defined in saved source |
| Pouring cameras | Wrist, head and overhead definitions; opt-in `ROBOTIS_POUR_ENABLE_CAMERAS=1`; default camera prims are passive placeholders |
| Pouring success | `never_success()` returns false for every environment; operator labels were intended for early demos |
| Pouring timing | Simulation dt 0.01 s, decimation 5, 30-second episode |

The pouring task is an incomplete scene/control baseline. Its saved configuration has no rewards and a placeholder success predicate. A task ID or successful particle creation does not establish a working pouring policy.

Source files are indexed in [details-manifest.json](../recovered/macbook-2026-10-28/details-manifest.json): `FFW_BG2.py`, `pick_place/joint_pos_env_cfg.py`, `pour_cup/ik_rel_env_cfg.py`, `pour_cup/pour_cup_env_cfg.py`, and `pour_cup/mdp/terminations.py`. The [original Quest runbook](../recovered/macbook-2026-10-28/source/AIWORKER_QUEST_XR_RUNBOOK.md) preserves exact historical launcher/recorder commands. Those target the old Dell container and include process-stop/setup operations; they are reference material, not a launch recipe for today's shared Ubuntu host.

## Evidence and limits

- **Current local verification:** all **83 CPU-only tests** in the recovered fluid tools passed on the Mac mini. These check configuration, geometry, parameter handling and asset filtering, not fluid simulation. [Test output](../evidence/recovered-fluid-tests.txt).
- **Raw historical logs:** the Dell empty-table tutorial created 1,370 particles and exported a stage; the blank tutorial created 5,060 particles, exported a stage and opened its gate at frame 120. The two-cup summary contains only a scene path and `particles=168`. These establish scene construction, not measured transfer or repeatable success.
- **Historical chat/runbook reports:** Quest teleoperation and baseline demonstration collection were used, a fluid sandbox UI was developed, and Isaac Sim 6.0.1 streaming was demonstrated on Brev. The July continuation returned to the baseline robot-task gate. The Brev simulator was stopped at the user's request; it was not contacted during this recovery.
- **Prior learned manipulation:** the June 22 [training analysis](../recovered/macbook-2026-10-28/source/outputs/aiworker_training_analysis/current_training_analysis_20260622.md) reports pick/place, pick/flip/place and bimanual handoff datasets and policies. These are historical reports of other tasks, not evidence of a learned liquid pour. That report also identifies an invalid earlier 7D/14D bimanual evaluation.
- **Fluid limitations:** the sandbox separates a 3,000-particle faucet cap from a 1,200-particle surface-preview cap after instability. Some scripts provide an appearance-only water preview. Particle count is not calibrated to a real delivered liquid mass.
- **No recovered evidence yet** establishes a successful end-to-end learned liquid pour or physical pour.

## September continuation and initial Ubuntu recovery inventory

The same project continued on home Ubuntu in September. Read-only inspection on October 28 confirmed:

| Item | Current observation |
| --- | --- |
| Access | Existing SSH alias `kfuUbuntu`, `kadir@192.168.68.60` |
| Original staged source | `/data/aiworker/src/aiworker`; `/home/kadir/aiworker` resolves to `/data/aiworker` |
| Sim 6 staged source | `/home/kadir/aiworker6/src/aiworker`, mounted as `/workspace/src/aiworker` |
| Container | `isaacsim6_home`, image `nvcr.io/nvidia/isaac-sim:6.0.1`, **stopped** (`Exited (255)`, reported four weeks ago) |
| Container GPU binding | Device `0`, the RTX 4090 |
| Native alternate install | `/home/kadir/isaac-sim`, VERSION `5.1.0-rc.19+release.26219.9c81211b.gl` |
| Native Isaac Lab | `/data/home/kadir/IsaacSimThings/IsaacLab`, VERSION `2.3.0`, commit `4f3e808ca75fd8ea07c20d188f067980eeab411f` |
| Scoped datasets/outputs | Both AIWorker workspaces' dataset/output directories contained no files |
| GPU use | RTX 4090 had an active humanoid-sitting Python process; RTX 4070 also present |

Evidence: [current inventory](../evidence/ubuntu-aiworker-inventory-2026-10-28.json). This is a point-in-time observation, not a reservation of either GPU. No `robotis_lab` checkout appeared in the two inspected AIWorker source directories; the stopped container's internal filesystem was not exhaustively searched.

Two original September 5 logs were copied from `/home/kadir/aiworker6/logs`. They contain `ISAACSIM6_SMOKE_APP_READY=1`, `ISAACSIM6_PHYSICS_CUBE_Z=0.219216`, and `ISAACSIM6_PHYSICS_OK=1`. [Log manifest](../recovered/ubuntu-2026-10-28/manifest.json). These prove startup and a small physics scene only. The final September chat said the next step was establishing a compatible ROBOTIS baseline on this machine. A standalone recovered-robot baseline has since been verified; the full ROBOTIS task remains unverified.

## Preserved artifacts

- [MacBook snapshot manifest](../recovered/macbook-2026-10-28/manifest.json): **106 selected source/note files**, 1,417,354 bytes, original paths, hashes, Git state and filename inventory. This is a scoped recovery, not a complete project backup.
- [Details manifest](../recovered/macbook-2026-10-28/details-manifest.json): **18 additional configuration/log files**, 57,582 bytes, with hashes.
- [Chat evidence](../recovered/macbook-2026-10-28/history.json): **70 selected final responses across five relevant chats**, with titles, IDs and timestamps. Full histories and private reasoning were not copied. The first snapshot's zero `final_count` fields came from a reader that missed compressed transcripts; this supplement corrects that limitation.
- [ROG backup scope](../recovered/macbook-2026-10-28/source/backups/rog-pairog2-20260630_180523/manifests/backup_scope.md): the old backup intentionally excluded datasets and checkpoint weights. Recovery has **not** restored trained weights or training datasets. Unrelated private backup files were not copied.
- Original project instructions are retained as `source/ORIGINAL_AGENTS.txt`, as historical reference rather than active project instructions.

## Historical recovery resume point

The standalone robot restart below has since been completed; current next actions and limits are in [STATUS.md](../STATUS.md) and [runtime/README.md](../runtime/README.md).

The immediate dependency is a compatible, isolated ROBOTIS baseline on a scheduled GPU. Preserve both existing Isaac versions, inspect the stopped container and locate/restore the appropriate `robotis_lab` checkout before choosing versions. Do not apply every old patch or upgrade dependencies together.

Then run **one** baseline environment, verify the composed robot/base and camera outputs, and save one teleoperated demonstration with the exact config/commit. Prepare the empty-container tilt-and-return with three resets, camera/state/action evidence and explicit dry-trial labels. The recovered pouring task defaults to 512 environments: override to one. Add measured fluid-transfer criteria after the dry baseline passes. Existing humanoid training retains priority; no GPU job was launched here.

Reproduce the completed local check:

```sh
cd /Users/kadir/linkedin/projects/aiworker-pouring
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s recovered/macbook-2026-10-28/source/scripts/fluid -p 'test_*.py' -v
```
