# Localization and settled-level correction

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
