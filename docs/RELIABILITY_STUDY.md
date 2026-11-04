# Temporal localization and spill repair

Completed November 4, 2026: **30/30 fresh tasks passed**, following 7/7 development passes on v17. The prior 27/30 benchmark is unchanged. The first development candidate passed 3/7 tasks; every failure and recording is retained.

## Fresh evaluation results

| Target | Strict successes | Final state |
|---|---:|---|
| Nearly full cup | 15/15 | 91.334–92.869% of rim height |
| Empty bottle | 15/15 | Zero source particles in every case |
| Whole task | 30/30 | Verified physical and visual grasp, transfer, correct stop, upright return and release |

All thirty reserved scenes ran once on the frozen candidate, without tuning, retries, restarts or expert fallback. Each transferred at least one nominal mL and had no final outside particles or sampled near-table loss. The primary score requires a sustained verified grasp, transfer, correct reason, complete return/release, the declared height/empty goal and zero measured loss. No-pour aborts cannot pass. [Verified results](../evidence/reliable-A/final-benchmark.json), [CSV](../evidence/reliable-A/final-benchmark.csv).

Thirteen tests had transient nonairborne outside classifications of one to three particles, all returning to zero loss; seventeen had none sampled. One particle represents 0.005832 nominal mL. The historical `ever_below_worktop_ml` field accumulates particles below cup-frame z=3.5 mm at **20 Hz observations**, while physics runs at **400 Hz**. Zero measured loss does not establish splash-free motion or exclude every between-sample excursion. Nominal particle-count volume is not calibrated physical volume. The separate legacy volume score was **16/30** (all fifteen empty cases and full case 23); the primary full goal is at least 90% surface height, not 90% nominal inventory. Scoring was unchanged. [Outcome analysis](../evidence/reliable-A/failure-analysis.json), [sampling review](../evidence/reliable-A/measurement-sampling-review.json).

Across the fifteen fresh full stops, the settled-height predictor's mean absolute error against final height was **0.518 percentage points**, with maximum overestimate **0.974 points**. These are decision-time measurements in this batch, not a general reliability guarantee. Material A, fixed cameras and millimeter-scale pose variation remain the evaluation scope; there is no physical-robot validation.

## Candidate and method

Study `A-reliable-20261103-v1`, prefix `C30_A_`, reserved seed `20261217`. All 30 scene specifications were generated deterministically before evaluation, with fifteen full and fifteen empty targets, disjoint from both earlier 30-case batches. Material A, vessel geometry, 1.8 mm particles, 400 Hz physics, deadlines and scoring remain unchanged. Cup position varies by ±8 mm in X/Y around (0.49, -0.15, 1.04) m; bottle offset varies by ±9 mm X / ±4 mm Y around (-0.12, 0) m. Full targets start with 85–100 nominal mL in the bottle and 3–14 in the cup; empty targets use 20–40 and 3–10 respectively. The whole-task deadline remains 40 simulated seconds.

Initial localization now holds the robot at home while collecting detections. It requires eight detections within one pixel, at least 60% support in a rolling twenty-frame window, and an anchor above the original confidence threshold. The median location drives grasp planning; isolated inconsistent estimates are excluded. The source support floor is 0.24, while the original source anchor threshold remains 0.343780; the cup threshold remains 0.631417. A two-second timeout aborts without grasping if support never arrives. This is temporal consensus, not a blanket lowering of the acceptance threshold.

Consensus calibration considered eighteen configurations on all thirty consumed parent windows. Selected settings gave maximum source error 0.623 mm and mean decision latency 0.35 s. All thirty synthetic neck occlusions were rejected. These are calibration results, not independent localization validation or real absent-object tests. A replacement-only CPU probe rejected three older initial views, so v17 retains both template families and chooses the higher family score. Native development and the reserved task evaluation measure subsequent behavior.

The settled-height head was refitted on frozen visual features with weighted ridge regression: `min_beta sum_i w_i (y_i - phi_i^T beta)^2 + lambda ||beta_nonbias||^2`. Labels are final settled height divided by rim height; episode weights avoid domination by long recordings and near-full samples receive threefold weight. Twenty feature/regularization candidates were compared. The selected head has sixteen coefficients and lambda 0.001. Data comprise 58 training and 18 validation episodes (2,109/634 frames), including completed parent recordings; every C30 recording is excluded. Neural cup/source encoders and motion-path coefficients remain unchanged. V17 additionally adds a normalized-gradient bottle template family from 24 consumed parent initial views while retaining the parent family, with six views excluded from new template construction; runtime still uses only RGB and fixed calibration. The method is supervised calibration within a visual servo/imitation/feedback controller, not reinforcement learning.

The initial RTX 4070 fit took 8.812 seconds; the cached v17 refit with the two template families took 4.716 seconds and retained identical level-regression validation metrics. Near-full validation MAE was 1.027 percentage points, p95 overestimate 1.637 points and maximum overestimate 2.130 points. The settled target is `0.90 + 0.016371106 + 0.005 = 0.921371106`. This margin is empirical, not a reliability guarantee. Some episodes were used for earlier encoder training, so validation is not independent for the entire perception stack.

The candidate reduces initial slow tilt from 10 to 5 degrees/second and holds further tilt when the RGB-derived flow estimate exceeds 18 nominal mL/s above estimated level 0.25. The existing tighter near-full hold, discrete top-ups, empty confirmation and upright return/release remain. V17 permits six of the same small top-ups (seven settling pauses including the initial pour), retaining the confidence target, pulse size and 40-second deadline. This flow estimate uses the existing visual derivative and nominal scale, not live particle counts. The motion adjustments are engineering controls; only the regression coefficients and consensus settings are fitted/calibrated.

## Development and verification

Development replays parent cases 19 and 26 (localization aborts), 29 (one-particle loss), and full/empty controls 1, 14, 0 and 10. Every attempted recording, failure and incomplete trial must remain identifiable. Round 0 completed 3/7 strict tasks: the three empty-bottle tasks passed, one localization attempt aborted, and three full pours reached 90.35–90.80% final height but returned `fill_unconfirmed` after exhausting four top-ups. All seven had zero final/near-table loss; the three full pours each had a transient one-particle outside reading. Seven HDF5 recordings and 28 sidecars match native SHA256 backups, and both task/wrist camera sheets were reviewed. [r0 outcomes](../evidence/reliable-A/round0-reliable-results.json), [r0 audit](../evidence/reliable-A/round0-development-integrity.json), [review](../evidence/reliable-A/round0-review-summary.json).

V17 checkpoint `40b5506bc4bceb51f1bf8ab8a5df93525933da0b03dea4c7e3fc66b63ba62101` passed all seven scenes under r1 names. Full targets ended at 90.638–92.675% rim height; all three short sources emptied, with complete return/release and zero final/near-table loss in every case. Four cases had transient outside classifications (one to three particles) that returned to zero. Both camera sheets were inspected; all fourteen r0/r1 HDF5 files and 56 sidecars match native checksums, with no partials. [r1 outcomes](../evidence/reliable-A/round1-reliable-results.json), [review](../evidence/reliable-A/round1-review-summary.json). Native sources match the local 22-file manifest. Fresh tests started only after candidate selection and ran without tuning or repeated attempts.

Twenty-six native CPU checks passed: fourteen grasp/stop, seven settled-level and five temporal-consensus checks. A separate local check verified thirty balanced, disjoint scene specifications with unchanged physical parameters. These checks are not simulated pouring successes.

The owned simulator was restarted only while idle during development, after checking its immutable identity. The launcher retained RTX 4070-only visibility, four CPU cores of quota on CPUs 16–23, CPU shares 256, nice 10, 16 GiB RAM/no extra swap and a guard that yields only this container if another 4070 workload appears. Before/intermediate/final snapshots verified the same container, caps, checkpoint, 22 sources and asset throughout fresh evaluation. No foreign job was signaled or reconfigured. A foreign 4090 process disappeared during earlier development for an unverified reason; these snapshots cannot establish uninterrupted execution of other workloads. At the final snapshot the queue was empty, recording closed, no runtime/learning error, guard active and approximately 51.7 GiB host RAM available. [Resource comparison](../evidence/reliable-A/runtime-comparison.json).

## Evidence and reproduction

Small reports live in `evidence/reliable-A/`; native files in `/workspace/learning/A-reliable-20261103-v1`, and local raw backups in `data/material-A-reliable/`. Images, videos, checkpoints and raw records remain outside Git. Frozen at `2026-11-04T08:33:11Z`: source commit `76c726ea0dfb17392d879959ca9ad844e7a2334e`, selected checkpoint `40b5506bc4bceb51f1bf8ab8a5df93525933da0b03dea4c7e3fc66b63ba62101`, scene plan `47346bfdc0bb7e85cf4d08778b9d07f021f967ae48afd3e8e6b7ed796f2b6074`. All 22 native/local runtime files match; zero spill-readiness allowance was used. [Frozen selection](../evidence/reliable-A/final-selection.json).

The native audit validated **44 HDF5 recordings / 19,697 transitions**, comprising fourteen development attempts and thirty fresh tests, plus **176 sidecars**, with no partial files. All 220 files match local backup SHA256 checksums, including the preserved unsuccessful development round. Maximum camera/state timestamp skew was 6.82e-7 seconds. All five fresh-test camera sheets were inspected: each shows task/wrist views at stop and final release. Selected frames from actual full-case 00 and empty-case 06 videos were also reviewed; the videos are 27.1 and 18.8 seconds at simulated-time speed. This visual sampling does not replace the trajectory/particle audit or certify continuous splash-free motion. [Integrity manifest](../evidence/reliable-A/recording-integrity.json), [review summary](../evidence/reliable-A/review-summary.json).

Completed verification commands, from native `/workspace/src` or the local repository respectively:

```sh
/isaac-sim/python.sh pour_reliable_report.py --require-thirty
/isaac-sim/python.sh audit_reliable_study.py integrity
/isaac-sim/python.sh audit_reliable_study.py review
python3 tools/summarize_reliable_benchmark.py
python3 tools/analyze_robust_outcomes.py --study reliable
```

The local verifier independently recomputed all thirty primary scores and checked the frozen identities, resource snapshots, data separation and all backup hashes. The analysis performs no fitting. These thirty cases are now observed evidence; any successor trained on them needs a new reserved evaluation. A next study could expand object poses or material appearance, retaining this frozen result unchanged.
