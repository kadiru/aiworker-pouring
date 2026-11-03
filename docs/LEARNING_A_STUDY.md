# Material A learning study — November 2, 2026

Collection, bounded GPU training and all six reserved native tests are complete. The final policy verified grasp in 5/6 tests but completed 0/6 pouring/stopping tasks; the zero-spill task target is not met. Material A is fixed for both vessels: light frost, opacity 0.08 and roughness 0.20, with the existing visual bands. These are simulator recordings, not real-robot or human teleoperation data.

## Demonstrations and admission

The frozen plan varies cup position by up to ±10 mm per horizontal axis and relative bottle position by up to ±12/6 mm. Source fills span 18–100 nominal mL; reference cups begin at 0, 8 or 14 mL. Six reserved tests use different intermediate placements and source/cup fills of 24/5, 90/11, 34/5, 100/11, 22/5 and 92/11. Seeds are 20261102 (original scenes), 20261103 (four crossed-fill training scenes) and 20261104 (corrective grasp prefixes). The original validation/test specifications were unchanged by either training-only extension.

Recorded 24 full scripted references, totaling 6,964 transitions. All passed the standard physical checks, 23 had exactly zero measured spill, and 16 met the stricter zero-spill plus ≥90%-full threshold. Several full-cup references stopped at 89.3–90.0% nominal brim; this is retained in the strict score. The first reference lost three particles (0.017496 nominal mL) and was excluded from fitting. Nineteen full training references and four full validation references were admitted after physical and camera review.

Eight additional physically disturbed approach/lift prefixes contain 65 transitions each. Six verified a stable zero-spill lift; two failed grasps were retained and excluded. These are partial skills, not full pours. Smooth arm disturbances of 0.25/0.40/0.55 degrees end at 2.4 simulated seconds; recordings end around 3.3 seconds. `actions` contains the actually issued disturbed commands; separate `teacher_actions` contains clean expert targets at the corresponding observation. Prefix full-task completion labels are zero. Final training therefore uses 19 full references plus six corrective prefixes, totaling 5,862 transitions, with 1,186 validation transitions. The 32 reference/prefix files total 7,484 recorded transitions.

Two 384×384 virtual task/wrist RGB cameras and measured joint states are recorded at 20 Hz simulated time, using JPEG95. Each transition includes paired observation, action and next observation with exact timestamp continuity. Camera matrices, metrics and supervision remain in separate fields. All admitted camera views were inspected. [Manifest](../evidence/learning-A/dataset-manifest.json), [collection audit](../evidence/learning-A/collection-audit.json), [backup index](../data/material-A-study/backup-index.json).

A collection build failure occurred when an initial 105 mL specification exceeded the bottle's fill lattice capacity after two saved episodes. It was corrected to 100 mL before fitting or evaluation. The original plan and interruption evidence are retained. Two successful calibration pours are excluded from the learning splits. Failed references, policy rollouts and incomplete files are not silently substituted for successful demonstrations.

## Learned controller and GPU use

Five bounded recurrent behavior-cloning candidates were fitted with native PyTorch 2.7.0+cu128 on the RTX 4070. Each fit ran 4,000 updates inside the already-owned simulator process, with physics/rendering paused; there was no competing CUDA training process. V1 used residual joint commands; v2 used absolute arm targets/shared finger closure; v3 added direct proprioceptive readouts and small sensor perturbations; v4 added corrective prefixes; v5 added ten-step action forecasts and overlapping forecast averaging. Earlier weights, curves, source and failures are preserved by version.

Final v5 has two spatial RGB encoders, a 192-unit GRU, direct measured joint features and 337,240 parameters. At 20 Hz it predicts ten future absolute joint targets (0.5 seconds). Overlapping forecasts decay over three steps and are filtered using its own predicted stop mode. A newly latched stop clears stale forecasts. Training-only ridge initialization uses admitted training examples; invalid future targets after recording/prefix ends are masked. Prefixes receive half the episode sampling. Validation windows are fixed for checkpoint comparisons.

Inference receives **only task RGB, wrist RGB, measured joint positions and velocities**. Object poses, particle volumes, scenario parameters and teacher actions never enter the actor. Auxiliary phase, stop, fill and completion heads use supervision during training. Joint rate bounds and native actuator limits still apply. The bottle remains a dynamic body held by actual finger contact, with gravity and no attachment constraint.

Final v5 selected update 1900 from 4,000 updates in 127.8 seconds, excluding image loading. Checkpoint SHA256 is `6b8287136f67bee224544c8c6c0beeb7fe4b1aaae3278da334d43d00765cc011`; [versioned weights](../evidence/learning-A/best-bc-v5.pt), [model/source/data provenance](../evidence/learning-A/model-summary-v5.json), [training history](../evidence/learning-A/training-history-bc-v5.json). Offline validation loss and image ablation are diagnostics, not task success.

Only GPU UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6` (RTX 4070) is assigned. The owned container retains four CPU cores of quota, CPUs 16–23, shares 256, nice 10, 16 GiB RAM/no extra swap and an immutable-ID guard that yields to foreign 4070 compute. No humanoid or sibling project jobs were changed.

## Native evaluation

All four final-v5 validation cases verified grasp, but **0/4 completed the task**. The held/tilted bottles stalled before transferring liquid. Earlier candidates also had no successful validation task; their results remain grouped by checkpoint rather than averaged into v5.

All six reserved test cases ran sequentially on the unchanged v5 checkpoint, with no further fitting or parameter changes. **5/6 verified grasp, 0/6 delivered liquid to the cup, 0/6 reached a learned stop decision, and 0/6 completed a zero-spill task** under either strict or operational criteria. Five held bottles stalled before flow. One missed grasp spilled 19.286424 nominal mL; final escaped-volume mean was 3.214404 nominal mL across six cases. Five zero-spill stalls are failures. Every missed grasp, spill and timeout remains in the results.

Evaluation ends at an eight-second grasp deadline or 32-second overall simulated horizon; these bounds terminate attempts and supply no actions. Test 00 ended at the grasp deadline; the other five reached the overall horizon. The policy made no fluid transfer in any final test (the at-least-1-mL count is also zero).

| Case | Initial bottle / cup, nominal mL | Verified grasp | Added to cup, nominal mL | Final outside, nominal mL | Completed |
| --- | ---: | :---: | ---: | ---: | :---: |
| A_test_00 | 24 / 5 | No | 0.000 | 19.286424 | No |
| A_test_01 | 90 / 11 | Yes | 0.000 | 0.000000 | No |
| A_test_02 | 34 / 5 | Yes | 0.000 | 0.000000 | No |
| A_test_03 | 100 / 11 | Yes | 0.000 | 0.000000 | No |
| A_test_04 | 22 / 5 | Yes | 0.000 | 0.000000 | No |
| A_test_05 | 92 / 11 | Yes | 0.000 | 0.000000 | No |

[Complete JSON](../evidence/learning-A/final-benchmark.json), [CSV](../evidence/learning-A/final-benchmark.csv), [all model attempts](../evidence/learning-A/study-results.json). Actual held-out camera recordings: [grasp/lift then stall](../evidence/learning-A/heldout-test01.mp4), [missed grasp/spill](../evidence/learning-A/heldout-test00.mp4).

The original strict score requires verified grasp, sustained bilateral contact while tilted, correct stop-reason label, upright set-down, policy completion, and zero escaped/below-worktop particles. Cup-full cases also require ≥90% nominal brim. A separately declared operational score evaluates the user's physical OR goal: the same grasp/set-down/completion/zero-spill conditions and either cup inventory ≥90% brim or bottle residual ≤max(0.25 mL, 1% initial source). Internal stop-reason classification is reported separately. No-motion or stalled zero-spill episodes do not count as task success.

Completion is the policy's own stop latch followed by open fingers and low measured joint velocity for ten control steps (0.5 seconds); actual set-down and fluid outcomes are independently scored from simulator state. Native particle inventory remains finite and conserved. Reported volumes use spacing-cubed nominal particle volume, quantum 0.005832 mL, not physical volume calibration. `ever_table_ml` is a below-worktop-plane proxy, not a contact-force measurement. The cup nominal brim is 72.8 mL. The simulation results do not establish real-robot performance.

## Reproduce and inspect

The [live lab](http://192.168.68.60:8212/) distinguishes **Expert demo** from **Learned policy** controls. Runtime POST `/control` commands are `learning_collect`, `learning_train`, `learning_val` and `learning_test`. A new test action queues six fresh rollouts; it is not a passive replay. Do not press it during a running benchmark.

Remote project: `/home/kadir/aiworker-pouring-4070`. Study: `learning/A-20261102-v1/`. Final-v5 native validation/test run: `outputs/fountain/20261102T232206Z/`. The current owned container ID is `1a6beecd207c817c25e030b0cb5f82f362fd0f913758f89a937de696c2b73f10`; verify its owner label and guard before any lifecycle action. Historical IDs in older status sections are superseded.

After all tests, run `docker exec -w /workspace/src aiworker_pouring_4070 nice -n 10 /isaac-sim/python.sh study_report.py` on Ubuntu, copy the resulting `study-results.json`, then run `python3 projects/aiworker-pouring/tools/summarize_a_benchmark.py` from the Mac workspace. The summarizer requires exactly six unique test cases for the final checkpoint/run. `export_pour_video.py EPISODE OUTPUT_DIR` exports synchronized native task/wrist frames using CPU only; local ffmpeg makes review clips. The prior detailed [implementation worklog](learning-A-worklog-20261102.md) is historical, not the current state.


The final checkpoint, actor source, data manifest and frozen plan match their recorded hashes. All ten final validation/test HDF5 backups match the native files byte-for-byte by SHA256: [evaluation backup index](../data/material-A-evaluation/backup-index.json). [Final runtime verification](../evidence/learning-A/runtime-final.json) confirms the original 4070-only caps and guard, an empty queue, inactive recording and no runtime/learning errors. The lab remains available for inspection with no fit in progress. No humanoid or sibling jobs were changed.

The next learning experiment should address closed-loop recovery from the observed held/tilted stalls using full-task corrective examples and longer temporal training. The six cases above are now used evaluation evidence. A later candidate requires fresh reserved scene seeds; repeating these cases would be a regression check, not a new untouched test set.
