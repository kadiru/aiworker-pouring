# Recover the July Isaac Sim project

**Recovery completed October 28, 2026.** Direct SSH found `/Users/kadirumacbook/Documents/aiworker`, the FFW-BG2 asset/configuration, Quest control and fluid/pouring source, and relevant saved Codex history. The September continuation on home Ubuntu was also inspected read-only. See [the recovered setup and evidence](RECOVERED_SETUP.md) for current findings.

This chat executes on **Kadir’s Mac mini (Apple M4)**. The user authorized SSH recovery and installed the required public key. [Connection details](SSH_ACCESS.md). No GPU process was started or stopped. The checklist below remains useful for runtime inspection; the old paste-in prompt is retained for reference and no longer requires user action.

## Prompt to paste into Codex on the MacBook Pro

> Recover my early-July 2026 AIWorker / ROBOTIS / FFW / Isaac Sim project. Search available chat titles and relevant local project files for AIWorker, ai_worker, FFW-BG2, FFW-SG2, Isaac Sim, Isaac Lab, cyclo_lab, pouring, Al-Jazari, and El-Jazari. This may have been a stationary-base AIWorker model. Work read-only. First inspect chat/project inventories, then targeted file searches; do not modify app databases. Find the exact project folder, chat title/ID if available, simulator and Isaac Lab versions, robot USD or URDF path and variant, base-fix setting, last runnable command, current script/controller, saved scenes, datasets/checkpoints and the last evidenced result/error. Distinguish what actually ran from suggestions in a chat. Identify whether the simulator ran locally or on another host. Return a concise handoff with paths and source evidence. Do not launch Isaac Sim, install/update packages, train, or disturb any running GPU jobs. My continuation project is on the Mac mini at /Users/kadir/linkedin/projects/aiworker-pouring.

That was a recovery request only. No new chat was created or messaged. The selected read-only SSH recovery succeeded, so there is no need to paste the prompt elsewhere.

## Minimum recovery handoff

| Item | Evidence to capture |
| --- | --- |
| Original work | Exact chat title, date range, project directory and host |
| Versions | Isaac Sim, Isaac Lab, Python, repository commits/submodules, container image tag/digest |
| Robot | USD/URDF path and hash, variant, articulation root, base fixed/free, joint names, gripper type |
| Scene | Stage path, meters-per-unit, up axis, gravity, timestep, solver, bottle/cup collision geometry |
| Control | Working launch command, action space, frames/units, joint order, control rate, gains |
| Perception | Camera prims/extrinsics/resolution/FPS, saved calibration and timestamp convention |
| Evidence | Video/log/error from the last run, datasets and policy/checkpoint paths, actual progress |

For a recovered Git checkout, read-only inventory commands include `git status --short`, `git rev-parse HEAD`, and `git submodule status`. Search the recovered directory with `rg --files -g '*.usd' -g '*.usda' -g '*.urdf' -g '*.py' -g '*.yaml' -g '*README*'`. Do not run a placeholder path or edit the original environment before the inventory is understood.

## Smallest simulation experiment after recovery

Use a project-owned copy/config with a single environment only when the shared GPU schedule allows it. Open the recovered robot and existing empty-bottle/cup scene. Verify scale, fixed base and reset state before motion. If those props do not exist, add simple rigid bodies with measured dimensions in the project copy; record their provenance. First rehearse a pre-grasped empty bottle: approach the cup, tilt, return upright, and stop. Keep the cup and base fixed. No invented world coordinates or guessed joint arrays.

Record a continuous camera clip, joint/action trace and the scene/config identity. Repeat three resets with identical initial conditions. Pass only if each dry cycle completes within 60 seconds, has no unintended contact/drop, keeps the bottle mouth above the cup opening during the intended pour interval, and returns to the declared upright rest. Inspect that last geometric condition manually in the clip; the trial logger does not compute it. Label the result **“Isaac Sim — scripted dry rehearsal”** or **“Isaac Sim — teleoperated dry rehearsal,”** according to the control source.

A tilted empty bottle is not fluid transfer. A visual stream, particles, or kinematic bottle attachment must be explicitly disclosed. Before claiming simulated pouring, establish what fluid model is actually running and how captured/spilled mass is measured. No liquid-physics validation is claimed by this plan.
