# Visual stopping and coordinated pouring

Completed November 3, 2026, in native Isaac Sim on the RTX 4070. **All six fresh autonomous tests finished: 4/6 complete zero-spill task successes under both scoring definitions.** This is simulation evidence; hardware transfer remains untested.

- **5/6 verified grasps, 5/6 liquid transfers, 5/6 completed return/set-down/release sequences.**
- All three bottle-empty cases passed with zero source residual and zero measured spill.
- One near-full case passed at **94.80% surface height**, 92.42% nominal inventory, with zero spill.
- One grasp failed: no bilateral contact or lift; **81.76464 nominal mL** ended outside the vessels. One held-bottle case stopped early and settled at **83.17%**, with zero spill. Neither is credited as a task success.
- No model, actor, scene, physics or configuration changes occurred during the six-test block. No expert takeover was enabled. The overall zero-spill task is **not yet reliable**.

## Six-case evidence

Volumes below are nominal particle-count volumes, not calibrated physical mL. Initial positions vary by millimeters, material A and physics remain fixed.

| Test | Source / cup initially | Final height | Source left | Outside | Result |
|---|---:|---:|---:|---:|---|
| 1 | 23 / 6 | 47.65% | 0.000 | 0.000 | Pass |
| 2 | 93 / 4 | 7.06% | 11.238 | 81.765 | Failed grasp |
| 3 | 37 / 10 | 70.84% | 0.000 | 0.000 | Pass |
| 4 | 89 / 13 | 83.17% | 44.446 | 0.000 | Early full stop / underfill |
| 5 | 19 / 2 | 35.80% | 0.000 | 0.000 | Pass |
| 6 | 97 / 9 | 94.80% | 38.713 | 0.000 | Pass |


[Machine-readable report](../evidence/stop-A/final-benchmark.json), [CSV](../evidence/stop-A/final-benchmark.csv), [every test's actual cameras](../evidence/stop-A/test-contact-sheet.jpg), [empty-source test video](../evidence/stop-A/autonomous-empty-test00.mp4), [near-full test video](../evidence/stop-A/autonomous-full-test05.mp4).

## What is learned, and what is programmed

This is **hierarchical supervised imitation with contextual movement primitives**. A retained visual behavior-cloning model grasps and lifts. A separately trained CNN estimates liquid surface height, source-empty probability and source visibility. A contextual trajectory imitates demonstrated pour joint targets. A programmed irreversible latch joins pouring to return, set-down, release, retreat and hold. Every completed terminal trace traversed those modes in that order, with no return to pouring.

The selected pour primitive fits, at each demonstrated phase:

`W = argmin ||XW − (q_expert − q_lift)||² + λ||W||²`.

`X` contains an intercept and normalized measured lift joints. CUDA fitting used **19 successful training paths** and **four separate validation paths**; validation selected λ=0.1 from0.1,1,10,100. Phase advances at demonstration-fitted30°/s initially and10°/s later, slowing/holding with visual fill/flow; supported phase ends at131°. The terminal primitive reverses this rollout's own measured history, using timing/warp/release profiles fitted from successful demonstrations.

The **152,589-parameter liquid detector** uses Huber-like fill regression and binary cross-entropy for empty/presence. The retained grasp/lift model has337,240 parameters. Latest bounded4070fit ran1,800 detector updates,1,600 updates of an earlier increment head, plus CUDA ridge path fitting in58.94s excluding loading. That37,383-parameter increment head is retained in the bundle for provenance but **inactive** in the final primitive actor. All CUDA fitting occurred within the existing simulator process.

Runtime inputs are task/wrist RGB, measured joints/velocities, the robot's own history and phase. Hardware FK estimates tilt from measured joints. Object poses, scenario fills, particle inventory and teacher commands are training/scoring information only. No bottle attachments or particle edits are used in policy evaluation. This is a fixed scene/material-A experiment with small pose variation.

## Goal and stopping criteria

Primary success requires verified sustained grasp, `(surface height >=90% rim OR source residual <=max(.25 nominal mL,1% initially))`, the correct stop reason, upright set-down/open-gripper completion, and **zero measured escaped particles**. Particle resolution is0.005832 nominal mL. Zero spill after no pour or an early stop is not task success. Inventory and worktop escape are sampled during native rollouts; this is not calibrated real-fluid validation.

Nominal inventory and visible surface level disagreed in the earlier data. **Before any new native run**, primary fullness was declared in measured surface-height terms; the original>=90% nominal-inventory score was retained separately. Both final six-case scores are4/6, although the final development full replay passes the surface goal and fails the old inventory goal.

The full latch uses predicted fill plus a short flow/runoff forecast, with two consecutive votes. The final minimum predicted-fill gate is0.80; the future target remains0.91. The empty latch requires probability>=0.97, source visibility>=0.7, proprioceptive tilt>=95°, low visual flow, and12 consecutive frames(0.6s). The learned source-volume regression is not used to decide empty. These are programmed operating rules on learned perceptions.

## Data quality and pre-test development

Perception began with4,541 training /2,801 validation triplets from prior recorded evidence. Unreliable high-height failed-render labels were masked; admitted references and low-height splash negatives were retained. Source labels are masked when the source has fallen out of view.122 cup-only low-height frames from the consumed first native validation provided corrective splash negatives; their source labels and actions were unused. Final train count4,663 triplets. Exact masks are in[quality](../evidence/stop-A/perception-label-quality.json) and[corrections](../evidence/stop-A/corrective-perception.json). Old consumed test images are perception-only evidence; none of the six new tests entered fitting.

Current offline cup MAE is6.894 percentage points, with5 false full calls among2,049 low-fill validation frames, including failure states. The older reference-only4.52-point/0-of385 metric uses a different mask. Empty detection at0.97 gives18/42 raw positive detections and29 raw false calls; after the offline active-pour/tilt/low-flow proxy gate,1/1 positive and0 false. **One gated positive is sparse calibration evidence.** An independently selected earlier4070-trained source head is combined with the current cup head; candidate hashes and caveats are saved.

All six development attempts are retained. The first V8 full case stopped around22% and spilled13.203648 nominal mL; its small-source case left.419904 and spilled.069984. V9 introduced spatial cup red-row features, splash negatives, the absolute-target pour primitive, and longer empty persistence. Its empty-source development passed. For the full-source lead bracket, gate0.82 gave95.45% settled surface and five escaped particles;0.78 gave88.11% and zero spill. The midpoint0.80 completed a native full-source development replay with zero spill and>=90% surface before testing. All learned weights and paths stayed bitwise identical during this operating-point calibration.

## Freeze, backups and runtime

Final checkpoint **b47569e3331b39818e3bd04975c7487d9c23ece1e8a51be42133243a1e938b35**, frozen **2026-11-03T10:17:45.315667Z** before the first fresh case. [Selection and all development attempts](../evidence/stop-A/final-selection.json). Frozen plan SHAa0f6acc52235335719db34ada7a4979596ff2fdbd2eb05f816abeabbb64d69e5; seeds20261112 development /20261113 test. Source/cup fills are23/6,93/4,37/10,89/13,19/2,97/9. Positions/fills/1.8mm/400Hz physics and A are unchanged from the predeclared plan. Isosurface render capacity was increased before native development; original plan and rejected fits are archived.

**12/12 HDF5 recordings passed integrity checks; all12 HDF5 and48 label/decision/parameter/metric sidecars match native SHA256 backups.** No partial file remains. [Backup index](../data/material-A-stop/backup-index.json). Both actual-camera videos are768×432 at10fps; empty-source14.7s, near-full11.9s. All six final outcomes were visually reviewed.

Owned container`eddff8172b2c63c14709a455183b39f969e77ab9cc31fe39997fd877703d1cf4`, owneraiworker-pouring. It sees only4070UUIDGPU-cde23916-c5fa-f18e-a343-127f732cc2f6, with4-core quota, CPUs16–23, shares256, nice10,16GiB/no extra swap, and immutable-ID guard. Final lab is idle, recording closed, queue empty, no runtime/learning error; **no training is left running**. Ten deployed/local source hashes and the original robot asset match; checkpoint and limits are unchanged across testing.

**Other-job limitation:** a4090 compute process(PID1383190) present before testing was absent at the end. This work sent no stop, signal, lifecycle or configuration commands to foreign jobs. Read-only kernel checks found no OOM or NVIDIA fault entries, and53.6GB RAM was available. Its exit reason is unverified; continued foreign-job execution is not claimed. [Runtime comparison](../evidence/stop-A/runtime-comparison.json), [system check](../evidence/stop-A/other-job-check.json).

Remote project`/home/kadir/aiworker-pouring-4070`; study`learning/A-stop-20261103-v1`; final native run`outputs/fountain/20261103T095357Z`. [Live lab](http://192.168.68.60:8212/). Native CPU audit commands:

```sh
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh stop_report.py --require-six
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh stop_integrity.py
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh stop_provenance.py
```

Local backup/report verification: `python3 tools/summarize_stop_benchmark.py` from this project. Do not rerun`stop_test` and describe those cases as the original reserved suite. The next improvement is grasp verification/retry and fill estimation normalized to the visible cup rim, with new reserved cases for subsequent evaluation.
