# Grasp confirmation and settled fill feedback

Native Isaac Sim study `A-guard-20261103-v1`, November 3, 2026. This follows the previous frozen six-case study, which achieved four complete zero-spill tasks: one missed grasp spilled, and one premature full stop left an 83.2% surface height. Those results remain unchanged in `VISUAL_STOP_STUDY.md`.

## Method and changes

The method is **visual servoing plus supervised imitation**. It combines calibrated image localization and inverse kinematics for grasping, a supervised cup-height CNN, a demonstration-fitted contextual movement primitive, and a programmed feedback/return sequence. This cycle does not train an end-to-end grasp network or use reinforcement learning.

1. A supervised gradient-template bank locates the bottle neck and cup rim in the task camera. Known camera, bench and robot geometry map image coordinates into a grasp target. The neck-to-base height transform is explicit and regression-tested. Bottle/cup positions are inferred from RGB at runtime.
2. Before enabling any pour, six consecutive 20 Hz observations must show a raised bottle neck in the expected image window, measured TCP rise above 35 mm, and obstruction of **both** driven gripper fingers. The obstruction threshold was fitted from successful demonstration grasps: 3.213 degrees short of the empty fully closed position. Failed confirmation triggers one lower/open/retreat/retry, then a safe failure.
3. A 96×96 image normalized around the detected cup rim supplies a CNN and an independent red-liquid geometry check. The CNN uses supervised particle-surface labels offline. Simulator liquid inventory, object poses and contact labels never enter runtime decisions.
4. The learned pour path is conditioned on measured lift joints and visually located cup XY. Coarse pouring forecasts runoff, then brakes over the cup at 80 degrees. The controller waits 1.2 simulated seconds, checks a stable level estimate, and either confirms full or resumes a small top-up. A full estimate with uncertain geometry cannot request more liquid. Each top-up is capped at 95 degrees, 0.1 seconds at the cap and 0.9 seconds total, with at most two top-ups. Full uses a predicted 94% target; **the physical success criterion remains 90%**.
5. Empty detection retains the previous supervised source detector, combined with measured-joint tilt, low visual flow and twelve confirming observations. A final full/empty decision latches irreversibly. Return follows the robot's own measured outbound phase envelope, excludes previous top-up peaks, then places, opens and retreats. This prevents a repeated tilt during return.

Runtime inputs: task/wrist RGB, joint positions/velocities, own history/phase, and fixed sensor/hardware/bench calibration. No expert fallback; the bottle moves through simulated gripper contact rather than an attachment or prescribed bottle pose. Generalization beyond this fixed calibrated fixture is untested.

## Training and brief mathematics

Training ran inside the existing guarded **RTX 4070** simulator process, with no second CUDA process. CPU data preparation and audits do not initialize CUDA.

- Cup CNN: **58,561 parameters**, **4,711 training / 2,687 validation images**. AdamW, 1,800 updates, selected step 400; the final cached fit took 11.29 seconds. The loss is weighted smooth L1: `min_theta Σ w_i Huber(f_theta(image_i) − height_i)`. Near-full and calm frames receive greater weight.
- Calm near-full validation: **74 frames**, mean absolute error **1.208 percentage points**, 90th-percentile absolute error **3.862 points**. Broad all-frame MAE is **6.471 points**. These masks differ; the smaller near-full metric is not an all-frame accuracy claim.
- Localization: 52 training / 30 validation initial frames; 16 source-neck and eight cup-rim templates. Maximum validation position errors: source **0.272 mm**, cup **0.499 mm**, within this calibrated simulated camera.
- Movement primitive: **19 training / four validation** successful original demonstration paths. Ridge regression solves `W = (XᵀX + Λ)⁻¹XᵀY`; the input is lift joints plus cup XY, with targets indexed by pour angle. Validation selects λ=0.001; Λ uses that diagonal penalty and a smaller intercept penalty.
- Objective at execution: complete a verified grasp and return/release while achieving `height/rim ≥ 0.90 OR residual ≤ max(0.25 nominal mL, 0.01 × initial)`, with the correct stop reason and **zero measured escaped particles**. Nominal mL are particle-count volumes, not calibrated physical volumes.

## Selection and evaluation protocol

All new `G_A_*` episodes are excluded from neural and motor fitting; controller thresholds are selected using development only. Earlier tests become consumed development evidence; they are never presented as fresh tests again. This study reserves seed **20261122**, six source/cup fills **26/4, 91/12, 39/8, 87/14, 20/3, 98/6 nominal mL**, with the same small position variation. Plan SHA256: `eae551d7eebf42a0f4deb26fae62805e0733c5e47fe3fcf429982c1467a8f10f`.

Two exact old failure scenes and two new validation scenes precede selection. The final checkpoint/config/source identity is frozen before the six reserved tests. Run each test once, without tuning during the block. Maximum 40 simulated seconds and a 12-second grasp deadline permit a retry/top-ups; the scoring thresholds and accepted liquid physics stay fixed. Material A, 1.8 mm spacing, 400 Hz physics and existing isosurface capacities remain unchanged.

Development failures remain evidence: an initial neck/base coordinate bug was caught before pouring; a broad template search could falsely reject a physically held bottle; a subsequent candidate completed all four grasps without spill but left one regression at 89.2% settled height. A rejected 96% prediction target caused a retained overflow failure because the model stayed near 94–95% even at the rim. The final correction uses a reachable 94% target, longer settling, and strict angle/time limits for relative top-ups. Native development exposes a larger bias than the original offline validation metric. Model tensors and fitted paths remained bitwise identical during these controller-only changes; actual fitting source hashes and final runtime source hashes are recorded separately.

## Results

The selected checkpoint passed all four development tasks with zero spill. The three full cases finished at 90.3–91.2% rim height; the short-source case emptied the bottle. Legacy nominal-inventory score: 1/4.

**All six reserved tests then ran once without policy changes:4/6 complete physical tasks, with zero measured spill in all six.** Six initial grasps were physically verified, five passed visual confirmation, five transferred liquid, and four completed the proper full/empty return/release sequence. A safe abort is not a successful pour. Legacy nominal-inventory score: **1/6**, retained separately.

| Test | Source/cup initial nominal mL | Final rim height | Source left nominal mL | Outcome |
|---|---:|---:|---:|---|
| G_A_test_00 |26/4|7.1%|25.999|Failed: visual confirmation rejected a physical grasp; retry localization failed |
| G_A_test_01 |91/12|91.1%|38.864|Pass: full, return/release, zero spill |
| G_A_test_02 |39/8|70.7%|0|Pass: empty, return/release, zero spill |
| G_A_test_03 |87/14|90.5%|37.366|Pass: full, return/release, zero spill |
| G_A_test_04 |20/3|37.9%|0|Failed: empty confidence stayed below .97; 40s timeout without return |
| G_A_test_05 |98/6|90.7%|40.252|Pass: full, return/release, zero spill |

All three full-cup cases passed; the short-bottle cases expose two remaining detection failures. Test00 had seven supporting image matches in 14 observations but no six-consecutive run. Test04's empty confidence remained around .94–.95 despite source0 and visible-bottle confidence >.99. Neither is counted as a zero-spill task success. The reliability objective is still unmet. [Machine-readable results](../evidence/guard-A/final-benchmark.json), [CSV](../evidence/guard-A/final-benchmark.csv), [all six camera views](../evidence/guard-A/test-contact-sheet.jpg).

Checkpoint SHA256: `f4ddb8b9f288ad2e5e8a9e6ac495f57a94c7413a0cdb4bbb040a89769bd36098`, frozen at 2026-11-03T13:56:40.556375Z. All 17 runtime/physics source hashes, checkpoint, container identity, resource caps and original robot asset stayed unchanged through testing. The separate 4090 process PID 1758861 remained present before/after; no foreign process was signaled or modified. [Isolation comparison](../evidence/guard-A/runtime-comparison.json).

All 19 completed native HDF5 recordings and 76 JSON sidecars match their local backups. One aborted development file retains native status `incomplete` and error `Stopped by user`; its original bytes/checksum are preserved and it is excluded from complete-task data. [Integrity manifest](../evidence/guard-A/recording-integrity.json). Actual completed videos: [full cup](../evidence/guard-A/autonomous-full-test01.mp4), [empty bottle](../evidence/guard-A/autonomous-empty-test02.mp4). Media/models/raw records remain external to Git.

The next repair uses consumed evidence from this batch and a separate fresh test seed; see [transparent-neck/empty confirmation follow-up](GRASP_VOTE_STUDY.md). No result from that follow-up is claimed here.

## Isolation and verification

Only container `aiworker_pouring_4070` with owner label `aiworker-pouring` is controlled. It is restricted to GPU UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, four-core CPU quota, CPUs 16–23, CPU shares 256, nice 10, and 16 GiB RAM/no extra swap. An immutable-container-ID priority guard yields the 4070 if a foreign compute process appears. No foreign job is stopped or modified.

Ten CPU regressions cover neck/base coordinates, false grasp confirmation, persistent positive confirmation, safe top-up selection and return without re-tipping. The existing 14 trial-log and five GPU-guard checks also pass. These CPU checks are separate from native task success.

Native commands, after required external runtime artifacts are restored:

```sh
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_guard_report.py --require-six
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_guard_audit.py integrity
```

Local backup/source verification: `python3 tools/summarize_guard_benchmark.py`. Native controls are `guard_train`, `guard_development`, and `guard_test`; send the last only after the development records close and `freeze_guard_suite.py` writes the selection record.
