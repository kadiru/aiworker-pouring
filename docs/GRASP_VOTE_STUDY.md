# Grasp, pouring and stop repair

Native Isaac Sim study `A-guard-20261103-v2`, November 3, 2026. **All six independent fresh tests passed the declared rim-height-or-empty task with zero measured spill.** Each test verified the grasp, transferred liquid, stopped for the correct reason, and completed upright set-down, release and retreat. Full cups finished at **90.7–91.0% rim height**; all three short bottles finished with **zero source particles**. This is a six-case result within the calibrated material-A fixture, not a broad reliability estimate or physical-robot verification.

The legacy criterion based on >=90% nominal particle inventory scores **3/6**. It remains unchanged and separate: the three full cups meet the measured surface-height goal while their nominal inventory is about 88%. Nominal mL are particle-count volumes, not calibrated physical volumes. Neither an underfill nor an aborted no-pour is credited merely for spilling nothing.

## What changed

The method is **visual servoing plus supervised imitation**, coordinated by a feedback state machine. A supervised template bank locates bottle/cup in RGB; calibrated IK approaches and lifts the free dynamic bottle. Six continuous physical checks require both gripper fingers to remain obstructed and measured TCP rise >35mm. Six supporting raised-neck matches within 12 observations tolerate brief image flicker; losing physical evidence clears the window. Verification is bounded at 1.25s, with one retry.

A trained, rim-normalized cup-height CNN and color-geometry check control coarse pour, brake, 1.2s settle and bounded top-ups. Full uses predicted height .94 with stability/agreement checks; the physical scoring threshold stays .90. Top-ups retain the 95deg cap, .1s cap hold, .9s total limit and maximum two attempts. A final stop latches; return follows the robot's own monotone joint-phase history so previous top-up peaks cannot tip the bottle again.

Empty retains the original .97/12-observation branch. A slower branch requires model score >=.90 for 24 observations, the learned 131deg phase cap with >=.25s dwell, measured tilt >=128deg, continued two-finger obstruction, visible-source score >=.7 and visual flow <1 nominal mL/s. It then uses the same return/release sequence. Runtime inputs are RGB, measured joints, own history/phase and fixed camera/robot/bench calibration. No live object/particle labels, object attachment, scripted bottle pose or expert fallback enter control.

## Training and brief mathematics

The cup CNN was retrained in the existing **RTX4070** simulator process: **58,561 parameters; 4,711 training / 2,687 validation images; 1,800 AdamW updates; selected step 400; 11.29s cached fit**. Loss: `min_theta sum_i w_i Huber(h_theta(image_i) - height_i)`. Calm near-full validation uses 74 frames: MAE 1.208 percentage points, p90 absolute error 3.862 points. Broad all-frame MAE 6.471 points is a different mask.

The pour primitive uses CUDA ridge imitation from 19 training/four validation successful demonstration paths: `W = (X^T X + Lambda)^(-1) X^T Y`, selected lambda .001. Context is measured lift joints plus visually located cup XY. The follow-up controller calibration retained cup/source tensors and all fitted paths **bitwise identical** to trained parent `f4ddb8b9f288ad2e5e8a9e6ac495f57a94c7413a0cdb4bbb040a89769bd36098`; it did not perform another neural fit. CPU calibration/audits initialize no CUDA context.

## Selection and preserved failures

The preceding frozen `G_A_*` round completed 4/6 tasks and had zero spill in all six; both failures remain in [its report](GRASP_AND_FILL_STUDY.md) and commit `9a4b8b1`. One physically held bottle was rejected by flickering image matches. Another emptied its source but stayed below the .97 empty gate until its 40s timeout.

Consumed-trace replay selected the temporal/slow-empty repair. On the missed-empty trace it would call at 12.305s, source 0, 1.5s after empty, with 90.1% bilateral contact up to the call. No eligible negative-at-cap frames existed; zero observed premature calls does not establish a false-positive rate. Three exact native regressions then passed: rejected grasp, missed empty, and good full-cup control. Full control finished 90.97% rim height; both short sources finished 0. Legacy regression score 2/3. [Calibration](../evidence/guard-B/vote-calibration.json).

The new six-case seed is **20261126**, with the same small source/cup position variation and source/cup fills 27/5,90/13,37/8,88/14,21/3,98/6. No `H_A_*` data entered neural/motor fitting or pre-test calibration. Model/configuration and 18 controller/physics hashes were frozen at **2026-11-03T15:17:56.265517+00:00**, source commit `00b1604a158a36398403855d83ae14b506936ca5`. No policy changes or tuning occurred during these six tests.

## Every fresh outcome

| Case | Source/cup initial nominal mL | Stop | Final rim height | Nominal inventory | Source left nominal mL | Spill |
|---|---:|---|---:|---:|---:|---:|
| H_A_test_00 | 27/5 | empty | 51.07% | 43.96% | 0.000 | 0.000 |
| H_A_test_01 | 90/13 | full | 90.88% | 87.76% | 39.109 | 0.000 |
| H_A_test_02 | 37/8 | empty | 68.13% | 61.81% | 0.000 | 0.000 |
| H_A_test_03 | 88/14 | full | 91.04% | 87.94% | 37.978 | 0.000 |
| H_A_test_04 | 21/3 | empty | 39.86% | 32.97% | 0.000 | 0.000 |
| H_A_test_05 | 98/6 | full | 90.72% | 87.54% | 40.270 | 0.000 |

Primary scoring requires sustained verified grasp, surface height >=90% rim OR source <=max(.25 nominal mL, 1% initial), correct reason, complete upright set-down/open-gripper sequence, zero final outside particles and zero ever-below-worktop particles. All six transferred at least 1 nominal mL. The three low-fill cups are successful because the source emptied; they are not treated as full cups. [JSON](../evidence/guard-B/final-benchmark.json), [CSV](../evidence/guard-B/final-benchmark.csv), [all task/wrist stop/final views](../evidence/guard-B/test-contact-sheet.jpg).

Checkpoint **`b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b`**; plan **`05a0bec06cf0b54b1ce110e61e0fd44617ea2bcb6e4f756d9235da0d6b52ec2a`**. All 18 runtime source hashes, checkpoint, container, original robot USD and resource caps match before/after. Material **A** and 1.8mm/400Hz physics remain fixed. The `guard-B` folder names this second repair round, not vessel material B.

## Evidence and isolation

All **nine native HDF5 recordings + 36 label/decision sidecars** match native SHA256 backups in `data/material-A-guard-vote/`; no partial files. Both recorded cameras were reviewed across all six outcomes. [Integrity](../evidence/guard-B/recording-integrity.json), [full-cup video](../evidence/guard-B/autonomous-full-test01.mp4), [empty-bottle video](../evidence/guard-B/autonomous-empty-test00.mp4). Prior 19 complete G_A recordings, 76 sidecars and one interrupted partial remain separately preserved. Recordings/models/media stay external to Git.

Only owned container `76ed304145f11503596a2996ee85af34d6c4c511b1760345047141a67c684cbc`, owner `aiworker-pouring`, was controlled. It used only 4070 UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, four-core quota, CPUs 16–23, shares 256, nice 10, 16GiB RAM/no extra swap, and an immutable-ID priority guard. Final lab idle, queue empty, recording closed, no runtime/learning error. All **33 CPU checks** pass: 14 grasp/stop, 14 trial-log and five GPU-priority guard checks.

The foreign 4090 process changed from PID 1841148 to 1920567 during the block; the old exit reason is **unverified**. No foreign process was signaled, stopped, restarted or configured. Read-only kernel checks found no OOM/NVIDIA fault entries. Do not claim uninterrupted foreign-process continuity from this evidence. [Runtime comparison](../evidence/guard-B/runtime-comparison.json), [read-only job check](../evidence/guard-B/other-job-check.json).

Reproduce the native report with `docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_guard_report.py --require-six --study vote`. Verify restored local backups/source with `python3 tools/summarize_guard_benchmark.py --study vote`. Required simulator/assets/checkpoints remain external; see [artifact guide](VERSION_CONTROL.md).
