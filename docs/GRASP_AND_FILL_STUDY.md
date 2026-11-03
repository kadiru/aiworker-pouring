# Grasp confirmation and settled fill feedback

Native Isaac Sim study `A-guard-20261103-v1`, November 3, 2026. This follows the previous frozen six-case study, which achieved four complete zero-spill tasks: one missed grasp spilled, and one premature full stop left an 83.2% surface height. Those results remain unchanged in `VISUAL_STOP_STUDY.md`.

## Method and changes

The method is **visual servoing plus supervised imitation**. It combines calibrated image localization and inverse kinematics for grasping, a supervised cup-height CNN, a demonstration-fitted contextual movement primitive, and a programmed feedback/return sequence. This cycle does not train an end-to-end grasp network or use reinforcement learning.

1. A supervised gradient-template bank locates the bottle neck and cup rim in the task camera. Known camera, bench and robot geometry map image coordinates into a grasp target. The neck-to-base height transform is explicit and regression-tested. Bottle/cup positions are inferred from RGB at runtime.
2. Before enabling any pour, six consecutive 20 Hz observations must show a raised bottle neck in the expected image window, measured TCP rise above 35 mm, and obstruction of **both** driven gripper fingers. The obstruction threshold was fitted from successful demonstration grasps: 3.213 degrees short of the empty fully closed position. Failed confirmation triggers one lower/open/retreat/retry, then a safe failure.
3. A 96×96 image normalized around the detected cup rim supplies a CNN and an independent red-liquid geometry check. The CNN uses supervised particle-surface labels offline. Simulator liquid inventory, object poses and contact labels never enter runtime decisions.
4. The learned pour path is conditioned on measured lift joints and visually located cup XY. Coarse pouring forecasts runoff, then brakes over the cup at 80 degrees. The controller waits 1.2 simulated seconds, checks a stable level estimate, and either confirms full or resumes a small top-up. A full estimate with uncertain geometry cannot request more liquid. Full uses a conservative predicted 96% target; **the physical success criterion remains 90%**.
5. Empty detection retains the previous supervised source detector, combined with measured-joint tilt, low visual flow and twelve confirming observations. A final full/empty decision latches irreversibly. Return follows the robot's own measured outbound phase envelope, excludes previous top-up peaks, then places, opens and retreats. This prevents a repeated tilt during return.

Runtime inputs: task/wrist RGB, joint positions/velocities, own history/phase, and fixed sensor/hardware/bench calibration. No expert fallback; the bottle moves through simulated gripper contact rather than an attachment or prescribed bottle pose. Generalization beyond this fixed calibrated fixture is untested.

## Training and brief mathematics

Training ran inside the existing guarded **RTX 4070** simulator process, with no second CUDA process. CPU data preparation and audits do not initialize CUDA.

- Cup images: **4,711 training / 2,687 validation**. AdamW, 1,800 updates, selected step 400; the final cached fit took 11.29 seconds. The loss is weighted smooth L1: `min_theta Σ w_i Huber(f_theta(image_i) − height_i)`. Near-full and calm frames receive greater weight.
- Calm near-full validation: **74 frames**, mean absolute error **1.208 percentage points**, 90th-percentile absolute error **3.862 points**. Broad all-frame MAE is **6.471 points**. These masks differ; the smaller near-full metric is not an all-frame accuracy claim.
- Localization: 52 training / 30 validation initial frames; 16 source-neck and eight cup-rim templates. Maximum validation position errors: source **0.272 mm**, cup **0.499 mm**, within this calibrated simulated camera.
- Movement primitive: **19 training / four validation** successful original demonstration paths. Ridge regression solves `W = (XᵀX + λI)⁻¹XᵀY`; the input is lift joints plus cup XY, with targets indexed by pour angle. Validation selects λ.
- Objective at execution: complete a verified grasp and return/release while achieving `height/rim ≥ 0.90 OR residual ≤ max(0.25 nominal mL, 0.01 × initial)`, with the correct stop reason and **zero measured escaped particles**. Nominal mL are particle-count volumes, not calibrated physical volumes.

## Selection and evaluation protocol

All new `G_A_*` episodes are excluded from fitting. Earlier tests become consumed development evidence; they are never presented as fresh tests again. This study reserves seed **20261122**, six source/cup fills **26/4, 91/12, 39/8, 87/14, 20/3, 98/6 nominal mL**, with the same small position variation. Plan SHA256: `eae551d7eebf42a0f4deb26fae62805e0733c5e47fe3fcf429982c1467a8f10f`.

Two exact old failure scenes and two new validation scenes precede selection. The final checkpoint/config/source identity is frozen before the six reserved tests. Run each test once, without tuning during the block. Maximum 40 simulated seconds and a 12-second grasp deadline permit a retry/top-ups; the scoring thresholds and accepted liquid physics stay fixed. Material A, 1.8 mm spacing, 400 Hz physics and existing isosurface capacities remain unchanged.

Development failures remain evidence: an initial neck/base coordinate bug was caught before pouring; a broad template search could falsely reject a physically held bottle; a subsequent candidate completed all four grasps without spill but left one regression at 89.2% settled height. The final controller correction increases the conservative prediction margin, waits longer for settling, and sets each trim goal relative to the settled level. Model tensors and fitted paths remained bitwise identical during these controller-only changes; actual fitting source hashes and final runtime source hashes are recorded separately.

## Results

Final development and the six reserved tests are in progress. No six-case success rate is claimed yet. Machine-readable selection, all attempts and final outcomes will be stored in `evidence/guard-A/`; complete HDF5 camera/joint/label records and any aborted development recordings remain external under `data/material-A-guard/` with checksum verification.

## Isolation and verification

Only container `aiworker_pouring_4070` with owner label `aiworker-pouring` is controlled. It is restricted to GPU UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, four-core CPU quota, CPUs 16–23, CPU shares 256, nice 10, and 16 GiB RAM/no extra swap. An immutable-container-ID priority guard yields the 4070 if a foreign compute process appears. No foreign job is stopped or modified.

Nine CPU regressions cover neck/base coordinates, false grasp confirmation, persistent positive confirmation, safe top-up selection and return without re-tipping. The existing 14 trial-log and five GPU-guard checks also pass. These CPU checks are separate from native task success.

Native commands, after required external runtime artifacts are restored:

```sh
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_guard_report.py --require-six
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_guard_audit.py integrity
```

Local backup/source verification: `python3 tools/summarize_guard_benchmark.py`. Native controls are `guard_train`, `guard_development`, and `guard_test`; send the last only after the development records close and `freeze_guard_suite.py` writes the selection record.
