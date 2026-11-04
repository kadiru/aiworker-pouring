# Localization and settled-level correction

## Selected candidate and current state

Frozen-batch interim:22/30complete,21passes. Test19 failed before grasp because source confidence.335436<.343780; cupconfidence.907903 passed. Four-frame CPU replay localizes within0.094–0.407mm of declared initial position, inside the expanded search bounds. The failure is retained and confidence thresholds remain unchanged during the batch. [Native failure](../evidence/settled-A/localization-failure-test19.json),[CPU diagnostic](../evidence/settled-A/localization-failure-test19-replay.json).

The candidate was frozen at2026-11-04T02:42:49Z (November3,18:42PST), sourcecommit27ba46e3bcebd71794d884b93f68ee9753a9a5e8. All30reserved cases were queued once. The read-only watcher verifies model and18runtime hashes every40seconds; outcomes are pending. [Frozen selection](../evidence/settled-A/final-selection.json), [resource snapshot](../evidence/settled-A/runtime-before-tests.json).

Development is complete for v15, checkpoint `18f0064ea9f1503488ba606a863a8047bf1f3cd8ff05dfb95a81b53e24121527`. All seven scenarios grasped, transferred, reached the appropriate full/empty goal and completed return/release. **Six of seven passed the unchanged strict zero-spill task.** The six full cases finished at 91.752–94.369% rim height; the empty control finished with zero source particles. All five failure scenes from the prior benchmark now pass. The full control `L30_A_dev_10_r4` lost two particles (0.011664 nominal mL); it remains a failure. An outside reading first appeared before its top-up, and particles were later observed below the table. Saved records do not identify individual particle trajectories, so the origin of both final escaped particles is unproven.

This candidate will be evaluated once on all 30 reserved cases to measure residual spill risk. The freeze command explicitly records a two-particle **development readiness allowance only**; it does not change any development or test score. The default freeze still rejects any strict failure. Checks against retained native outcomes reject underfill, missed grasp and the older eight-particle spill, while the two-particle case remains labeled failed. [Readiness checks](../evidence/settled-A/readiness-checks.json), [round4 outcomes](../evidence/settled-A/round4-settled-results.json).

The final predictor uses a frozen CNN plus 14 vertical color bands, 16 ridge coefficients, lambda 0.1, 37 training/12 validation episodes and 1,301/398 frames. Near-full validation MAE is 1.068 percentage points; p95 overestimate is 2.321 points. The selected target is 0.928209. Cached refitting took 1.112 seconds on the guarded RTX 4070; neural encoders and learned motor primitives remain unchanged. The v15 controller uses at most four short 35-degree/s pulses, immediately brakes at `min(110,max(95,previous_peak+0.25))` degrees, and evaluates settled predictions after each 1.2-second pause. Moving-image predictions cannot terminate a pulse. Twenty-one targeted CPU checks pass.

All 18 complete development recordings, 72 sidecars, three interrupted partials and three partial sidecars match native SHA256 backups. Both round4 task/wrist stop/final camera sheets were inspected. Failures and interrupted attempts from earlier rounds remain archived. None of the reserved 30 tests has started at this selection checkpoint.

The owned container is `47342c463a7c0edcb913a2191bf11bbade77faa20e424825ab1581337fb2fb27`, with 4070-only access, four-core quota, CPUs16–23, shares256, nice10, 16GiB/no extra swap and the immutable-ID GPU-priority guard. A foreign 4090 process changed before the round3 restart for an unverified reason; no foreign process or configuration was modified by this work. Do not interpret snapshots as proof of uninterrupted foreign-job execution.

Freeze after source commit: `AIWORKER_SOURCE_COMMIT=<commit> /isaac-sim/python.sh freeze_settled_study.py --document-known-spill-particles 2`, then POST `settled_test` once. No model/controller/plan tuning is allowed during the 30 cases. Run the final report, native recording audit, backup verification and camera review before reporting completion. The sections below retain the chronological development history; their earlier configurations and pending states are superseded by this selected-candidate record.

Authorized November3,2026: fix localization/settled-level prediction, retrain on the guarded4070, run30fresh tests, commit and push. MaterialA and the existing calibrated workspace/physics remain fixed. All earlier failures remain recorded in the [prior30-case report](ROBUSTNESS_A_30.md).

## Implemented correction and actual fit

Source localization bounds are now the union of the old bounds and projections of all16corners of the declared placement domain, with6pixel margin. Source x coverage expands from114–173 to98–174. Template kernels and confidence thresholds are unchanged. On36consumed initial views, both vessels were accepted in36/36, maximum source error0.743mm and cup error0.544mm. This is an offline localization check; native grasps still require physical and temporal image confirmation.

A five-coefficient supervised ridge regressor predicts final settled surface height from normalized-crop red-shape features and the existing cupCNN output. The objective is an episode-balanced weighted sum of squared prediction error plus lambda times coefficient norm squared. Lambda0.001 was selected from12feature/regularization candidates on held-out episodes. The old cup/source encoders, template kernels, demonstrated motor paths and empty gates remain fixed; this is a new fitted prediction component, not a new end-to-end motor policy or reinforcement learning.

Data:35previously consumed H_A/T30_A episodes,27training/eight validation,1,070/307frames. Frames are admitted only after cup particle inventory has stabilized to its final value; the target is final settled surface height. The crop uses the policy's own image-localized cup center. Train/validation are split by episode. FutureL30_A records are explicitly excluded. Ground-truth liquid labels are training/scoring data only; runtime inputs remain RGB, measured joints, own history and fixed calibration.

Actual fit in the existing4070 simulator process took4.831seconds. Near-full validation MAE0.709percentage points, p95absolute error1.093points, maximum overestimate0.903points across167near-full validation frames. The selected stopping target is predicted settled height0.915, derived from the unchanged physical goal0.90 plus validation overestimate allowance and margin. These are offline metrics, not autonomous success rates.

The controller evaluates six recent settled predictions, requires stability and agreement with the independent visual geometry estimate, and latches after three full votes. Coarse braking retains its old instantaneous predictor. Bounded top-ups use the new settled predictor, retain95degree maximum tilt and0.1second cap dwell/0.9second total pulse limits, with at most two pulses. Upright return/release and empty detection are retained.

Initial candidate SHA256:17010a756b59751744a57b6cd665bdef1d87d3a624b33d9675a8a5c5f0b6564f. Parent:b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b. [Training/model summary](../evidence/settled-A/model-summary.json), [data manifest](../evidence/settled-A/data-manifest.json), [localization validation](../evidence/settled-A/localization-validation.json).

## Development and reserved evaluation

Study`A-settled-20261103-v1`, prefix`L30_A_`, new reserved seed20261203. Seven development cases reproduce all five prior failures plus the successful highest-full and low-source controls. All development attempts are retained with unique round suffixes. Resolve development failures before freezing; do not tune on the reserved batch.

The reserved30cases contain15near-full and15bottle-empty targets, shuffled once. Same cup XY±8mm, source X±9mm/Y±4mm, full sources85–100nominalmL/cups3–14 and short sources20–40/cups3–10. MaterialA,1.8mm/400Hz physics, geometry, cameras,40second deadline and12second grasp deadline remain fixed. Primary height>=90%or source-empty, verified grasp/transfer/completed return and zero final outside/ever-below-worktop particles; legacy nominal-volume and transient nonairborne outside readings are reported separately. Zero spill without completing the task is a failure.

After development: commit the selected source, freeze model/plan/source hashes, run all30once without tuning, preserve every outcome, audit and back up native recordings and metadata, review all camera sheets, then commit/push the final evidence. Final counts are pending.18CPU checks passed before native development, including underfill/noisy-evidence rejection and test-scene disjointness.

## Isolation and reproduction

Only owned container b66b71ee3071919e82210760c909974d6e7e91247b84e92671b2e492ae80639f was restarted to load the new code. It retains4070UUIDGPU-cde23916-c5fa-f18e-a343-127f732cc2f6, four-core quota, CPUs16–23, shares256,nice10,16GiB/no extra swap and its immutable-ID priority guard. Foreign4090PID2630921 was observed and not modified. The prior19runtime files were checksum-archived before deployment. [Resource snapshot](../evidence/settled-A/runtime-before-development.json).

Commands: POST`settled_train`, then`settled_development` to the existing `/control` endpoint while idle. Native report:`/isaac-sim/python.sh pour_settled_report.py`. After all seven current-candidate development cases pass, `AIWORKER_SOURCE_COMMIT=<commit> /isaac-sim/python.sh freeze_settled_study.py`, then POST`settled_test`. Final report:`pour_settled_report.py --require-thirty`; CPU audit:`audit_settled_study.py integrity` and`review`. Local backup verification:`python3 tools/summarize_settled_benchmark.py`.

Use prior revision b2c7600 to run the previous robustness study's source-bound verification. Its artifacts and25/30score remain unchanged; current controller files intentionally differ.

## Development revision before reserved tests

Round0case03 grasped/returned with zero final spill and90.691%height, but correctly withheld the target0.915settled prediction:both95degree top-ups produced no flow, leading to `fill_unconfirmed`. The assistant stopped case08at4.55simseconds after diagnosing the motion limit; its90-transition partial and the complete failure are preserved. Five queued cases never started. None of the30reserved tests started. The initial model/summary remains archived as candidate-r0-17010a75.

Round1uses bounded adaptive top-ups:cap=min(110,max(95,previous pouring peak+1))degrees,35deg/s approach then4deg/s creep,0.1s cap dwell/1.5s total pulse, maximum two pulses. It retains the fitted five-coefficient predictor and threshold0.915; the repeated cached fit took1.067s with identical validation metrics. CandidateSHA1741c5ea37b18f4dfa63246553e23644e7bef9634e7308c823df08cfcbfb8004.19CPU checks pass. All seven development cases are now being evaluated with this candidate; their native outcomes are pending.


## Expanded high-fill calibration — development round2

Round1's first case spilled6.281064nominalmL after two top-ups; it is a failure. The predictor trained only on near-full tails up to92.27% and stayed near91.3% at the newly reached98.4% instantaneous height. The assistant stopped the remaining development queue; the next partial and candidate are retained. No fresh test was used.

The new admission list adds14 compatible high-fill prior recordings from A/R_A/S_A studies, all zero-final-spill, same material/rig/physics. Their final settled targets reach98.02%. Train37episodes/1301frames, validation12/398; these are consumed development/calibration data, and some were used in earlier encoder training. Final L30 tests remain the independent evaluation. Recovery policy logs can be shorter than camera recordings; only their fixed image-localized cup center is used, while labels are aligned against full samples by next_time. Other old recordings are localized from their initial RGB frame with the saved bank. No true object position supplies the crop.

Frozen CNN plus14 vertical red-color bands gives a16-coefficient selected ridge model, lambda.1, chosen among20candidates. Actual4070fit5.656seconds; near-full MAE1.068points,p95absolute/overestimate2.321points,maxoverestimate2.853points. Target=.90+.023208899+.005=.928208899. CPU replay of the failed r1 high plateau now gives96.7–97.0%, high enough to withhold another top-up. This is not a completed native trial. Seven round2 cases are running; no reserved tests yet.


## Top-up gate correction — round3 running

Round2 completed6/7primary tasks; all7had zero final outside/below-table loss. Case29 now grasped, but finalheight89.485%and `fill_unconfirmed` remain a failure. Its top-ups braked at97.5/102.75degrees on RGB changes before reaching the prior106degree pouring angle; total added only0.472392nominalmL. Other cases exposed larger surges with the old+1degree allowance. Round2 failures/partials, model, report and camera review are preserved.

Feedback braking now requires phase>=adaptive cap minus0.25degree. Cap=min(110,max(95,previous peak+0.5))degrees; capdwell0.05seconds,total1.5seconds,max two top-ups. Twenty CPU checks include the actual premature-braking angles. The same49episode ridge fit was rerun on4070 in1.119seconds, with identical errors and unchanged neural weights. NewmodelSHA a570bb051e4d005c914dbe9bc834a6ef30bfb28d86d1810949b7b6e4ed2baaa4. Seven development scenes are being rerun, prioritizing29; the reserved30scene plan is unchanged and unstarted.


## Pulse-and-settle control — round4 running

Round3 repaired29(96.091%height,zero final spill), but03overflowed(final0.046656nominalmL,everbelowworktop0.029160). A lower cap with slow creep increased the duration of flow. The two completed attempts and stopped08partial are retained; the r3 camera sheet visibly shows the spilled bead.

The v15controller separates measurement from motion. During top-ups it ramps35deg/s to min(110,max(95,previous peak+0.25))degrees and immediately brakes; there is no creep/cap dwell, and a1.5second timeout remains. Full/underfull decisions use the learned settled model after each1.2second pause, never its out-of-distribution moving/flowing predictions. At most four pulses fit inside the unchanged40second trial limit.21CPUchecks pass. Actual4070cached refit1.112s, same16coefficient predictor and data. Model18f0064ea9f1503488ba606a863a8047bf1f3cd8ff05dfb95a81b53e24121527. Native round4 prioritizes03and29; no fresh tests have started.
