# Thirty-case material-A robustness evaluation

**Completed November 3, 2026: 25/30 complete-task successes (83.3%).** The frozen controller passed all 15 bottle-empty cases and 10/15 near-full cases. Four pours stopped slightly short; one case rejected initial bottle localization before moving. All 30 attempts remain in the denominator. No retraining, threshold adjustment, material change or repeat attempt occurred during this evaluation.

| Measurement | Result |
|---|---:|
| Complete rim-height-or-empty tasks | 25/30 |
| Expected near-full tasks | 10/15 |
| Expected bottle-empty tasks | 15/15; all ended with zero source particles |
| Verified physical and visual grasps | 29/30 |
| Liquid transfers of at least 1 nominal mL | 29/30 |
| Complete upright return, set-down, release and retreat | 29/30 |
| Zero final outside and ever-below-worktop particles | 30/30 |
| Legacy nominal-inventory task score | 15/30 |

Successful near-full cups settled at **90.12–92.27% rim height**. The legacy score uses a different quantity: poured full-case inventories were 86.26–89.37% of nominal brim inventory, so none passed its 90% inventory cutoff. Its 15 passes are the empty-bottle cases. Height and nominal volume percentages are not interchangeable. Nominal mL are particle-count units, not calibrated physical volumes.

The declared spill criterion is zero final outside particles and zero particles ever classified below the worktop. **Seven runs briefly classified 1–3 nonairborne particles outside the vessels**, peaking at 0.017496 nominal mL. Those readings returned to zero; no below-worktop particles were recorded. Only 23/30 runs had no such sampled transient, including the no-pour abort. This does not establish zero particle excursions at every instant. The no-pour abort and four underfills are failures despite zero final spill.

## Failures and next correction

| Reserved case | Failure | Settled height |
|---|---|---:|
| T30_A_test_03 | Early full stop | 89.529% |
| T30_A_test_08 | Early full stop | 89.905% |
| T30_A_test_20 | Early full stop | 89.697% |
| T30_A_test_27 | Early full stop | 89.719% |
| T30_A_test_29 | Initial bottle localization rejected; no grasp/pour | 12.303%, starting liquid only |

For the four actual underfilled pours, the image estimate exceeded simultaneous measured surface height by 2.47–2.74 percentage points. After stopping, measured height fell another 2.34–2.42 points while cup particle inventory stayed constant. These observations support correcting the estimate of **final settled height** before latching full, rather than crediting the higher surface observed during settling. The data distinguish prediction error from later surface change; they do not identify a unique physical cause of that change.

For case29, offline CPU replay of four saved observations reproduced bottle scores 0.216–0.236, below the frozen 0.343780 threshold; cup scores 0.880–0.895 passed its 0.631417 threshold. The measured bottle neck projects to (111.176,141.581), outside the source search bounds x=114–173, y=118–153. This exposes inadequate localization coverage at a combination of allowed position offsets. The runtime recorded `localization_unconfirmed`; numeric scores are explicitly offline recomputation, not original GPU logs. `policy.done=true` denotes completion of its abort behavior, not completion of the pouring task.

The [failure analysis](../evidence/robust-A/failure-analysis.json) identifies 14 actual full-pour episodes for settled-height calibration and the rejected initial-view episode for localization repair. No new model has been fitted. Next: expand and validate localization coverage, calibrate settled-level stopping/top-ups on separate development scenes, then reserve new seeds for a successor benchmark. These 30 cases become observed development evidence for any successor; never present their reuse as fresh testing. Keep material A until those failures are repaired before adding B/C.

## Frozen protocol and provenance

Study `A-robust-20261103-v1`, prefix `T30_A_`, seed **20261130**. Thirty randomized cases were generated once: 15 expected near-full and 15 expected bottle-empty, shuffled before execution. Full-source starts were 85–100 nominal mL with cup 3–14; short-source starts 20–40 with cup 3–10. Cup XY varied by ±8 mm, source X by ±9 mm and Y by ±4 mm. This tests a small calibrated workspace, not broad manipulation or physical transfer.

The unchanged controller combines image-template visual servo grasping, a supervised cup-height CNN, a demonstration-fitted contextual ridge pour trajectory, and calibrated temporal/physical stop gates. Runtime inputs are RGB, measured joints, own motion history and fixed calibration. No live simulator object/particle labels or expert fallback enter control. The existing model and its training history are described in the [prior six-case study](GRASP_VOTE_STUDY.md); this round is evaluation only.

- Checkpoint SHA256: `b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b`, byte-identical to the prior six-case controller.
- Plan SHA256: `9035a10b7edd75db35246cb65f0a9f5693d2e409194d92c54630a6abd8885c3a`.
- Frozen source commit: `1a5560362abb640371680f1575f22cbe62c57f1f`.
- Frozen at 2026-11-03 19:30:35 UTC (11:30:35 PST); native run `20261103T192830Z`.
- All 19 runtime/scheduler/study hashes match before and after. Seventeen inherited controller/model/physics files match the prior six-case benchmark; the runner only adds the reserved-case queue command.
- Material A, 1.8 mm particle spacing, 400 Hz physics, cameras, geometry, 40 s deadline and 12 s grasp deadline stayed fixed.

Primary success requires sustained verified grasp, measured final surface height >=90% rim or source <=max(0.25 nominal mL,1% initial source), the appropriate stop reason, completed upright return/release and the declared zero-spill criterion. Final scoring, not the looser intermediate `eligible` field, determines success. All cases ran once. No test was interrupted or silently replaced.

## Evidence and resource isolation

[JSON](../evidence/robust-A/final-benchmark.json), [CSV](../evidence/robust-A/final-benchmark.csv), [failure measurements](../evidence/robust-A/failure-analysis.json), [localization replay](../evidence/robust-A/localization-analysis.json), [recording integrity](../evidence/robust-A/recording-integrity.json), [resource comparison](../evidence/robust-A/runtime-comparison.json).

All **30 HDF5 files +120 JSON sidecars** are validated and SHA256-matched to local backups under `data/material-A-robust/`: 8,346 transitions, 800,457,915 bytes including sidecars, maximum camera/state skew 0.350 microseconds, no partial recordings. Technical recording validity is separate from task success. Trial execution totaled 8,992.41 wall seconds, about 2.50 hours, excluding startup and audits.

All five camera sheets were inspected: [00–05](../evidence/robust-A/camera-review-01.jpg), [06–11](../evidence/robust-A/camera-review-02.jpg), [12–17](../evidence/robust-A/camera-review-03.jpg), [18–23](../evidence/robust-A/camera-review-04.jpg), [24–29](../evidence/robust-A/camera-review-05.jpg). Each includes task/wrist views at stop and final state. Actual recording videos: [successful full](../evidence/robust-A/autonomous-full-10.mp4), [empty bottle](../evidence/robust-A/autonomous-empty-00.mp4), [underfill](../evidence/robust-A/autonomous-underfill-03.mp4). These media and raw records are external artifacts, not GitHub-hosted files. Small reports and hashes are versioned.

Owned container `f783d7a6d1416de3a7b48419a12e3cffa4aba5cff189707ddb2fd1c3393a941b` stayed on RTX4070 UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, with four-core quota, CPUs16–23, shares256, nice10,16 GiB RAM/no extra swap and the immutable-container-ID priority guard. Before, intermediate and final compute snapshots show only the owned4070 process; no foreign job was signaled, stopped, restarted or configured. These snapshots are not a continuous audit of all host processes. The lab has an empty queue and closed recording, no runtime/learning error, and retains the final failed case for inspection.

## Reproduce the audit

On Ubuntu, within the existing owned container:

```sh
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_robust_report.py --require-thirty
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh audit_robust_study.py integrity
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh audit_robust_study.py review
docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh audit_robust_localization.py
```

After copying the reports and recordings to their documented local locations:

```sh
python3 tools/summarize_robust_benchmark.py
python3 tools/analyze_robust_outcomes.py
```

The plan generator passed four CPU checks, including the real saved scene schema. The final audit verifies all source/model/asset hashes, all native backups, all reserved names and absence of expert fallback. The `robust_test` queue command only submits unstarted reserved cases; it should now have nothing to submit. Use a new study and new reserved seed for future evaluation.
