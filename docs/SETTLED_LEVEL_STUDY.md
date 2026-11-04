# Localization and settled-level correction — 30 fresh tests

Completed November 3, 2026 (America/Los_Angeles). **27/30 complete-task successes (90.0%)** after RTX 4070 fitting and a frozen, once-only evaluation. Two cases rejected source localization before grasping; one completed the pour but lost one particle. **Reliable localization and zero spill are not yet solved.** [Verified results](../evidence/settled-A/final-benchmark.json), [all thirty CSV rows](../evidence/settled-A/final-benchmark.csv), [failure analysis](../evidence/settled-A/failure-analysis.json).

## Measured outcomes

| Measure | Result |
|---|---|
| Whole-task success with unchanged zero-spill requirement | 27/30 |
| Near-full tasks | 13/15 strict successes: one localization abort, one spill |
| Empty-source tasks | 14/15 strict successes: one localization abort |
| Verified physical/visual grasp, transfer and completed return/release | 28/30 each |
| Executed near-full pours reaching at least 90% rim height | 14/14; final heights 91.989–95.463%, mean 93.502% |
| Executed empty-source pours | 14/14 ended with zero source particles |
| Runs with zero final outside and near-table particle loss | 29/30, including the two no-pour aborts |
| Legacy nominal-inventory success score | 23/30, retained separately |
| Fresh camera/action transitions | 8,642; all thirty recordings valid |

No executed full pour stopped below the 90% rim-height goal. The 13 zero-spill full successes finished at 91.989–94.925%; the highest full cup, 95.463%, belongs to the spill failure. A no-pour abort is never credited as task success merely because it has no spill.

Twelve cases had sampled nonairborne particles outside the vessels: eleven returned to zero final/near-table loss; test 29 retained one lost particle. The peak was 13 particles, or 0.075816 nominal mL, in test 29. Its final loss was one particle, **0.005832 nominal mL**. Only 18/30 had no sampled nonairborne outside reading, including the two aborts. This is not splash-free performance. Volumes are particle count × spacing³, not calibrated physical mL. [Review statistics](../evidence/settled-A/review-summary.json).

Primary scoring requires sustained verified grasp, visual confirmation, transfer, the correct full/empty stop, completed upright return/release, and zero final outside particles plus none detected below cup-frame z=3.5 mm. Full means final measured rim-height fraction ≥0.90; empty means source ≤max(0.25 nominal mL,1% of its initial charge). The historical field `ever_below_worktop_ml` uses that 3.5 mm threshold; its predicate was not changed. The local audit independently recomputed all thirty whole-task scores from the saved fields. The legacy ≥90% nominal brim-inventory score is a different measure; it must not be described as physical fill height.

## What changed and what was learned

Localization search bounds now cover projections of all 16 corners of the declared placement domain, with a six-pixel margin, unioned with the old bounds. Source bounds are `[98,118,174,156]`; cup bounds `[234,201,275,228]`. Template kernels and acceptance thresholds remain unchanged. Offline checks accepted both vessels in 36/36 consumed initial views, with maximum source/cup position errors 0.743/0.544 mm. This fixed the previous search-bound exclusion, but the fresh confidence failures below show that it did not solve all localization failures. [Localization validation](../evidence/settled-A/localization-validation.json).

The learning method is **supervised ridge regression on frozen image features**, integrated with the existing visual servo, demonstration-fitted motion primitives and feedback state machine. It is not reinforcement learning or a newly trained end-to-end motor policy. The selected head has 16 coefficients: bias, existing cup-CNN output and 14 vertical red-color bands. Neural cup/source weights, template kernels, motor primitives and empty-confirmation gates were retained.

The fitted objective is `min_β Σ_i w_i(y_i − φ_iᵀβ)² + 0.1 ||β_nonbias||²`, where `y` is final settled height/rim height. It is solved as weighted ridge least squares on the 4070; episode weights prevent long recordings dominating, and targets above 0.8 receive threefold weight. Twenty feature/regularization candidates were compared. Data: **37 training / 12 validation episodes, 1,301/398 frames**; frames are admitted after inventory stabilizes, and crops use image-localized cup positions. All L30 episodes are excluded from fitting. Some calibration episodes had been used in earlier encoder training; these offline errors are not an independent test of the entire perception stack. The new thirty-case batch is the separate evaluation. [Data manifest](../evidence/settled-A/data-manifest.json), [model summary](../evidence/settled-A/model-summary.json).

Near-full validation MAE was 1.068 percentage points; p95 overestimate 2.321 points, max 2.853. The target is `0.90 + 0.023208899 + 0.005 = 0.928208899`, an empirical margin, not a statistical guarantee. Initial expanded fitting took 5.656 s on the RTX 4070; the final cached refit took 1.112 s with identical coefficients/validation metrics. On the 14 fresh executed full pours, the instantaneous settled-head prediction at the stop decision overestimated final height by 1.094–2.973 points, MAE 1.901 points. These fresh measurements remain reported even though all 14 reached the height goal.

V15 uses at most four short top-ups. Each ramps at 35°/s to `min(110,max(95,previous_peak+0.25))°` and brakes immediately; there is no slow creep or cap dwell, with a 1.5 s timeout. Decisions use six recent predictions after a 1.2 s settling pause: 25th-percentile estimate, standard deviation <0.01, agreement with visual geometry <0.10, sufficient red evidence and three full votes. Moving-image predictions cannot end a pulse. Existing coarse braking, empty detection and monotone return/release are retained. Twenty-one targeted CPU checks passed. Runtime inputs are RGB, measured joints, own history and fixed camera/robot/bench calibration; object/particle truth is used only for training labels and scoring.

## Retained failures

| Fresh case | Observation | Consequence |
|---|---|---|
| `L30_A_test_19` | First source score 0.335436 <0.343780; cup score 0.907903 passed | Aborted before grasp/pour; full cup goal unattempted |
| `L30_A_test_26` | First source score 0.324122 <0.343780; cup score 0.911089 passed | Aborted before grasp/pour; source remained 32.000184 nominal mL |
| `L30_A_test_29` | Correct grasp/full stop/return, but one particle escaped | Strict zero-spill failure despite 95.463% final height |

Both rejected source positions are inside the expanded search bounds. Their initial errors against declared scene placement were 0.094 and 0.181 mm. CPU replay of all 21 saved observations per abort found one later above-threshold frame for test 19 and three for test 26; those estimates were accurate. Test 19 also had two rejected low-confidence outliers about 15 mm off, so simply lowering the threshold is not established as a fix. The four-frame previews alone do not capture those outliers. No threshold was changed and neither case was rerun. [All-frame replay](../evidence/settled-A/localization-failure-all-frames.json), [test 19](../evidence/settled-A/localization-failure-test19.json), [test 26](../evidence/settled-A/localization-failure-test26.json).

Test 29's first outside/near-table reading was at 9.105 s, during the initial braking phase, before its top-up. Camera review shows a red bead beside the cup. The final outside particle is below the table/workspace. Aggregate samples do not reconstruct identities, so they do not prove that the final particle is the same one first observed. The failure remains counted without relaxing the zero-spill criterion. [Spill timeline](../evidence/settled-A/spill-failure-test29.json), [actual failed-cycle video](../evidence/settled-A/autonomous-spill-test29.mp4).

## Selection and preserved development

| Round | Completed strict successes | Retained issue |
|---|---|---|
| r0 | 0/1 | Nonflow top-ups; `fill_unconfirmed` at 90.691% height |
| r1 | 0/1 | Narrow predictor failed at high fill; 6.281064 nominal mL final spill |
| r2 | 6/7 | All zero final spill; one underfill at 89.485% |
| r3 | 1/2 | One full case lost 8 particles; lower cap with slow creep increased flow duration |
| r4, selected v15 | 6/7 | All grasp/stop/return conditions passed; full control lost 2 particles |

All five failure scenes from the previous benchmark passed the selected development round. V15 was nevertheless selected with the documented two-particle spill failure in a control case, to measure residual risk honestly in the reserved batch. The explicit `--document-known-spill-particles 2` option affected development readiness only; failed labels and strict test scoring remained unchanged. Three interrupted development recordings are retained; the assistant stopped those queues. Native attributes say “Stopped by user”, and separate interruption notes identify the actual actor. [Selection](../evidence/settled-A/final-selection.json), [readiness checks](../evidence/settled-A/readiness-checks.json), [round4](../evidence/settled-A/round4-settled-results.json).

Study `A-settled-20261103-v1`, reserved seed 20261203, fifteen full/fifteen empty cases shuffled once. Cup XY ±8 mm, source X ±9 mm/Y ±4 mm; full sources 85–100 nominal mL/cups 3–14, short sources 20–40/cups 3–10. Material A, 1.8 mm particles, 400 Hz physics, geometry, cameras and deadlines remain fixed. Frozen at 2026-11-04T02:42:49Z from source commit `27ba46e3bcebd71794d884b93f68ee9753a9a5e8`; checkpoint `18f0064ea9f1503488ba606a863a8047bf1f3cd8ff05dfb95a81b53e24121527`, plan `d30146de48db51711ec3a45ce3ba52541b16450b0159822e348344925d8ec8ea`. All 30 ran once without fitting, tuning, repeated test attempts or simulator restarts. The previous 25/30 result is retained separately; the different reserved cases do not establish a paired improvement or broad reliability.

## Evidence, isolation and reproduction

All 48 complete HDF5s + 192 sidecars +three interrupted partials +three partial sidecars match native SHA256 backups under `data/material-A-settled/`. Archived development hashes also match, preserving every earlier failed attempt. The fresh 30 contain 8,642 transitions; all completed attempts contain 14,618. All five task/wrist stop/final camera sheets and the localization/spill diagnostics were inspected. [Integrity manifest](../evidence/settled-A/recording-integrity.json), [review summary](../evidence/settled-A/review-summary.json), [first full video](../evidence/settled-A/autonomous-full-test01.mp4), [first empty video](../evidence/settled-A/autonomous-empty-test00.mp4). Media, weights and raw recordings remain external to Git.

Only the owned 4070 container was used: `47342c463a7c0edcb913a2191bf11bbade77faa20e424825ab1581337fb2fb27`, four-core quota, CPUs 16–23, shares 256, nice 10, 16 GiB/no extra swap and its immutable-ID priority guard. Before/intermediate/final snapshots agree on the container, caps, model, 18 runtime hashes and original robot asset. No foreign job was signaled or reconfigured. Foreign 4090 PIDs changed during the work; cause unverified, and snapshots do not prove uninterrupted foreign execution. A read-only kernel query covering the batch found no matching OOM/Xid entries. The lab is idle with an empty queue, closed recording and no runtime/learning errors. [Resource comparison](../evidence/settled-A/runtime-comparison.json), [kernel query](../evidence/settled-A/kernel-check-after-tests.json).

Regenerate native reports without running simulation: `pour_settled_report.py --require-thirty`, then `audit_settled_study.py integrity` and `audit_settled_study.py review` using `/isaac-sim/python.sh` in `/workspace/src` of the owned container. Local verification: `python3 tools/summarize_settled_benchmark.py`; analysis: `python3 tools/analyze_robust_outcomes.py --study settled`. Restore external artifacts/dependencies as documented in [VERSION_CONTROL](VERSION_CONTROL.md). The final selection intentionally prevents refitting or adding development trials to this study.

Next work is a separate candidate: validate multi-frame initial localization with positive and negative examples, and reduce the initial-flow splash/loss observed in test 29. Preserve this 27/30 benchmark, use observed cases as development data, and reserve new scenes before evaluating any successor. Material B/C and hardware transfer remain untested here.
