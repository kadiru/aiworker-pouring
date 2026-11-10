# Independent color and moving-view calibration

Active study: `A-scale-dynamic-20261109-v1`, on the immutable owned RTX 4070 container. This successor preserves the completed [static size study](LEVEL_SIZE_STUDY.md), including its nine consumed test episodes. Those test images do not enter this study's fitting or model selection.

The task is to estimate drink height in cups with height and diameter scales in `{0.9, 1.0, 1.1}` while a bottle moves nearby. Cup dimensions remain declared inputs. Material A and the original bottle size are retained. Authored kinematic bottle/hand fixtures test perception; they are not learned grasps or autonomous pours.

## Method and gates

Primary RGB and independent red-mask models regress normalized physical surface height. Each has 58,770 trainable parameters. Supervision comes from recorded particle state; crops use image-localized cup coordinates. Equal-size/equal-episode sampling minimizes squared prediction error with AdamW. The color model does not receive primary-model predictions or weights. Their agreement is an acceptance check, not a training target. This is supervised perception learning, not reinforcement learning or end-to-end motor learning.

Unchanged development limits include localization below 3 mm; per-size level MAE below 2 percentage points; near-full maximum error below 3.5 points and maximum overestimate below 2.5 points; primary/color disagreement below 3.5 points; CPU/GPU and shared-inference differences below 0.0001 fraction. Static and moving groups must pass separately. Missing color must produce rejection, not a guessed empty cup.

The high-fill fixtures target 95.6% height using a bounded 0.75 nominal-mL/s native particle emitter, followed by physical settling. Required measured coverage is 94–98%. At most five lost particles remains the user's practical task tolerance; calibration labels currently require zero measured loss and a stable inventory plateau.

Moving fixtures start upright, tilt and translate, then return. Low-cup views use a 20 nominal-mL colored bottle and a 65-degree maximum angle; high-cup views use an empty bottle and an 82-degree maximum. Translation is transverse to the tilt plane. Full-resolution cup-detail images are retained. Actual bottle movement and containment are measured independently; projected/planned movement alone is not acceptance evidence.

## Preserved unsuccessful candidates

All three initial fits used only the existing 126 training / 18 validation episodes on the 4070. None is accepted or activated for autonomy.

| Candidate | Worst-size validation MAE | Near-full max error | Max primary/color disagreement |
|---|---:|---:|---:|
| Physical-feature ridge/RBF r0 | 2.644 pp | 2.052 pp | 4.120 pp |
| Compact-feature fit r1 | 3.112 pp | 2.162 pp | 4.353 pp |
| Independent color-mask CNN r2 | 2.477 pp | 1.259 pp | 6.143 pp |

Reports: [r0](../evidence/dynamic-A/color-r0.json), [r1](../evidence/dynamic-A/color-r1.json), [r2](../evidence/dynamic-A/color-mask-r2.json). These results motivate 18 additional low-fill training scenes at existing training positions. Validation identities and accuracy limits are not changed to accommodate the failures.

## Current measured evidence

The three bounding high-fill pilots pass recording integrity, stable-label and zero-loss checks. Measured median levels are 95.6598%, 95.5368% and 94.6471%. All twelve final camera views were actually inspected on three sheets. This is selected-frame review, not full playback. [Pilot audit](../evidence/dynamic-A/seed_pilot-r0-audit.json), [review](../evidence/dynamic-A/seed_pilot-r0-manual-review.json).

All nine parked high-fill layouts, eighteen low-fill training layouts, and proposed moving training/validation paths pass sampled CPU geometry screening. All 18 reserved layouts also pass the same geometry preflight; no reserved images have been collected. Actual moving-view accuracy remains unverified at this checkpoint. Nine native CPU motion/emitter checks and five synthetic acceptance-gate checks pass. Synthetic checks do not count as simulated task successes.

All nine high-fill scenes now pass and all 36 final camera frames were inspected; twelve of eighteen low-fill training scenes are complete with zero loss. [Nine-scene summary](../evidence/dynamic-A/high-fill-summary.json). Moving collection, the updated fit, independent evaluation and any autonomous pours remain pending. The active parent robot checkpoint is unchanged. No resized autonomous acceptance has been issued.

A read-only CPU diagnostic of the retained models on the nine reviewed high-fill training recordings confirms all nine cup locations, with maximum error **1.0671 mm**. The retained primary/color maximum absolute errors are **3.4425 / 5.4770 percentage points**; maximum pairwise disagreement is only **2.7684 points**. Agreement alone therefore does not establish level accuracy. This diagnostic accepts no model and does not substitute for the forthcoming 4070 validation. [Diagnostic](../evidence/dynamic-A/retained-high-cpu-probe.json).

## Operation and artifacts

Native root: `/workspace/learning/A-scale-dynamic-20261109-v1`; Ubuntu host root: `/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1`. Small reports are copied to `evidence/dynamic-A/`; weights, camera sheets, seeds and HDF5 recordings stay outside Git. Unchanged baseline plan/audit/locator files are copied natively from the completed parent study and are already versioned under `evidence/scale-A/`.

`watch_dynamic_stage.py` runs a finite, once-only calibration queue. `check_dynamic_stage.py` audits records and measured seeds. `finalize_dynamic_protocol.py` requires actual camera review before binding high seeds to moving scenes. `dynamic_level_data.py` admits reviewed whole episodes and checks localization before fitting; moving labels are matched to each frame's timestamp. `fit_dynamic_pair.py` trains independent RGB/color regressors and retains unsuccessful candidates. `pour_dynamic_study.py` requires frozen development acceptance before reserved evaluation, and separate explicit all-gate acceptance before autonomy.

The owned container keeps 4070-only GPU access, a four-core CPU quota on CPUs 16–23, shares 256, and 16 GiB memory/swap. Its priority guard remains active. Only the idle owned simulator was restarted to load eight calibration support files. No foreign process was signaled or reconfigured; snapshots do not prove uninterrupted operation of other workloads.


### Current continuation commands

Run native CPU helpers through `/isaac-sim/python.sh` with `CUDA_VISIBLE_DEVICES=` inside the owned container. Host observers and the deployment/component wrappers run with host `python3`. Keep collection and GPU fitting sequential.

1. Let `run_dynamic_collection.py seed_remaining low_train` finish; inspect all resulting camera sheets and record their exact audit/image hashes in `*-r0-manual-review.json`.
2. Run `finalize_dynamic_protocol.py` only after all nine high seeds have passed actual review. It preserves protocol v2 and binds measured seeds into the predeclared moving cases.
3. Run `preflight_dynamic_geometry.py --part validation`, then `run_dynamic_collection.py validation`. Audit and actually review all moving views before admitting them.
4. Run `prepare_dynamic_fit.py --round r3`, then host `run_dynamic_component.py --helper fit_dynamic_pair.py --label pair-fit-r3 --round r3`. This checks localization before fitting. R3 trains on the original 126 episodes plus 27 new static episodes and selects against the original 18 plus 18 new moving validation episodes.
5. If development accuracy fails, reserved cases and autonomy stay unused. R4 support can add the already-declared 18 moving training scenes after native audit/review; it must retain all R3 failures and unchanged validation identities/limits.
6. Only a separately frozen component selection permits the 18 fresh reserved moving cases. Autonomous pouring additionally requires all-size robot motion clearance, runtime integration/parity and explicit all-gate acceptance. No command above grants that acceptance.
