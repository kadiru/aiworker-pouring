# Thirty-case material-A robustness evaluation

Authorized November 3, 2026. Evaluate the current controller without retraining or changing thresholds during the batch. Prior six-case benchmark:6/6 rim-height/empty successes,3/6 legacy nominal-volume successes; those results stay separate.

Reserved study `A-robust-20261103-v1`, prefix `T30_A_`, seed **20261130**. Thirty cases are generated once:15 expected near-full and15 expected bottle-empty, in shuffled order. Full-source starts are85–100nominal mL with cup3–14; short-source starts20–40 with cup3–10. Cup XY varies by±8mm, source X±9mm/Y±4mm. These are new cases within the existing calibrated workspace, not an expanded workspace or new vessel material.

Checkpoint remains byte-identical **`b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b`**. Seventeen inherited Actor/model/physics dependencies must match the previous frozen benchmark exactly. The runner adds only a queue command to submit reserved cases; its new hash and the study module are frozen before execution. Material A,1.8mm/400Hz physics, cameras, contact geometry,40s deadline and12s grasp deadline stay fixed. No expert fallback.

Score every case on sustained grasp, correct full/empty stop, complete upright return/release and zero measured escaped particles. Report both **>=90%rim height OR source<=max(.25nominal mL,1%initial)** and the unchanged legacy **>=90%nominal inventory** goal. Keep failures and incomplete records; zero spill from no pour is not a task success. Additional peak non-airborne outside volume is recorded separately.

The guarded4070 retains four-core quota, CPUs16–23, shares256, nice10 and16GiB/no extra swap. No foreign job may be stopped or reconfigured. If the priority guard yields the4070, preserve the interruption and only queue cases that never started; never overwrite/retry a reserved attempt silently.

Native command after freezing: POST `{"command":"robust_test"}` to the existing `/control` endpoint. The command queues only unstarted reserved cases. Summary: `docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh pour_robust_report.py --require-thirty`. Raw cameras/labels/models stay external to Git; small hashes/reports/source are versioned.

Status: prepared, not started. Final counts and camera/backup checks will be added after all30 outcomes are recorded. Any failure analysis precedes a separate training round; no changes to material B/C in this evaluation.
