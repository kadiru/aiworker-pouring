# Independent color and moving-view calibration

Active study: `A-scale-dynamic-20261109-v1`, on the immutable owned RTX 4070 container. This successor preserves the completed [static size study](LEVEL_SIZE_STUDY.md), including its nine consumed test episodes. Those test images do not enter this study's fitting or model selection.

The task is to estimate drink height in cups with height and diameter scales in `{0.9, 1.0, 1.1}` while a bottle moves nearby. Cup dimensions remain declared inputs. Material A and the original bottle size are retained. Authored kinematic bottle/hand fixtures test perception; they are not learned grasps or autonomous pours.

## Method and gates

Primary RGB and independent red-mask models regress normalized physical surface height. Each has 58,770 trainable parameters. Supervision comes from recorded particle state; crops use image-localized cup coordinates. Equal-size/equal-episode sampling minimizes squared prediction error with AdamW. The color model does not receive primary-model predictions or weights. Their agreement is an acceptance check, not a training target. This is supervised perception learning, not reinforcement learning or end-to-end motor learning.

Unchanged development limits include localization below 3 mm; per-size level MAE below 2 percentage points; near-full maximum error below 3.5 points and maximum overestimate below 2.5 points; primary/color disagreement below 3.5 points; CPU/GPU and shared-inference differences below 0.0001 fraction. Static and moving groups must pass separately. Missing color must produce rejection, not a guessed empty cup.

The high-fill fixtures target 95.6% height using a bounded 0.75 nominal-mL/s native particle emitter, followed by physical settling. Required measured coverage is 94–98%. At most five lost particles remains the user's practical task tolerance; calibration labels currently require zero measured loss and a stable inventory plateau.

Moving fixtures start upright, tilt and translate, then return. Non-high-cup views use a 20 nominal-mL colored bottle and a 65-degree maximum angle; high-cup views use an empty bottle and an 82-degree maximum. The nine training cases named `_low` preserve source seed cutoffs of 0.22 (five cases), 0.50 (two) and 0.78 (two); they are not uniformly low-filled. Training targets use measured height, never the name or initialization cutoff. Moving validation/reserved non-high fixtures use their declared low seed. Translation is transverse to the tilt plane. Full-resolution cup-detail images are retained. Actual bottle movement and containment are measured independently; projected/planned movement alone is not acceptance evidence.

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

All nine parked high-fill layouts, eighteen low-fill training layouts, and proposed moving training/validation paths pass sampled CPU geometry screening. All 18 reserved layouts also pass the same geometry preflight; no reserved images have been collected. Moving localization now has a measured failed gate, described below. Nine native CPU motion/emitter checks and five synthetic acceptance-gate checks pass. Synthetic checks do not count as simulated task successes.

All nine high-fill scenes pass and all 36 final camera frames were inspected. All eighteen low-fill training scenes pass at 26.2482–28.2414% measured height with zero loss; all 72 final camera frames were inspected. [Nine-scene summary](../evidence/dynamic-A/high-fill-summary.json). All eighteen moving validation scenes pass integrity, stable-label, motion and zero-loss checks. All 180 selected final/motion camera frames were actually inspected. Minimum measured motion is 45.903 mm translation and 64.991 degrees rotation; cup inventory stays constant. Independent evaluation and autonomous pours remain unused. The active parent robot checkpoint is unchanged. No resized autonomous acceptance has been issued.

A read-only CPU diagnostic of the retained models on the nine reviewed high-fill training recordings confirms all nine cup locations, with maximum error **1.0671 mm**. The retained primary/color maximum absolute errors are **3.4425 / 5.4770 percentage points**; maximum pairwise disagreement is only **2.7684 points**. Agreement alone therefore does not establish level accuracy. This diagnostic accepts no model and does not substitute for the forthcoming 4070 validation. [Diagnostic](../evidence/dynamic-A/retained-high-cpu-probe.json).

## Moving localization blocker and repair

The first combined attempt, r3, stops before any level/color optimizer update: **7/18 moving validation locations fail** (three unconfirmed, four errors of 3.592–4.462 mm). All 27 new static locations pass. [Rejected gate](../evidence/dynamic-A/localization-r3-rejected.json), [zero-update attempt](../evidence/dynamic-A/pair-r3-aborted.json). All 27 frozen inputs are archived natively under `pair-r3-frozen-inputs/`; the old checkpoint is not overwritten.

All eighteen predeclared moving training scenes pass integrity/stability/motion and zero-loss checks, with all 180 selected final/motion camera frames actually reviewed. Across all 63 new calibration scenes, 468 selected frames were reviewed; these are calibration fixtures, not autonomous successes. The separate `rim-d0` repair attempted to fine-tune the native 256-pixel rim crop CNN with its coarse detector frozen, whole-episode splits, Gaussian heatmap cross-entropy plus 0.02 smooth-L1 coordinate loss. It rechecks old static and new moving frames, temporal confirmation, synthetic absence separation and CPU/GPU parity. No gate is relaxed. It stopped on a non-finite heatmap loss before any checkpoint; optimizer update count was not recorded. All frozen inputs and the log are retained. The read-only diagnosis finds 86/15 train/validation frames outside the coarse crop (2,943/1,188 checked frames). Two worst native frames were actually inspected and retain visible cups. `coarse-d1` refits the full-image detector with focal heatmap plus 2× smooth-L1 offset loss, including authored absent-cup masks. Its 24-native-pixel crop bound is only an internal prerequisite: full fine-rim localization must still pass the unchanged 3 mm, temporal, absence and parity requirements. No failed frame is discarded. Fine-rim heatmap normalization now uses stable softmax and checks crop coverage before any optimizer update. Coarse d1 completed 100 epochs / 18,400 updates of 42,947 parameters in 489.62 s. Its selected crop errors are 19.817/15.088 native pixels, but absence margin -0.40931 fails >0.015; the candidate remains unaccepted. Coarse d2 starts from that retained candidate, balances static/moving groups equally, and uses 50% absent-cup samples. The masks, validation identities and gates are unchanged.

Moving level evaluation was expanded from the draft stride of two to **every recorded 20 Hz frame from 0.5 through 2.5 seconds**, before the first moving fit or error evaluation. Scene identities, physical motion and accuracy limits are unchanged. The r3 frame policy is frozen in its protocol.



Coarse d2 passes after 120 epochs / 22,080 updates, selecting epoch 120: crop errors 5.288/14.854 native pixels, absence margin 0.05949, CPU/GPU difference 0.02507 native pixels. This is only coarse-crop acceptance. Fine d1 then fails the unchanged 3 mm gate after 22,080 updates, despite 207/207 temporal confirmations: selected validation maximum 9.546 mm and training maximum 8.234 mm. Fine d2 retains every frame and old static check, balances static/moving training groups equally and starts from retained d1. It completes 160 epochs / 29,440 updates in 308.15 seconds, selecting epoch 105: train/validation maxima **2.480 / 9.073 mm**. All 207 temporal confirmations, 0.00746-pixel CPU/GPU difference and 0.00209-pixel shared-path difference pass, but full localization fails. No new level/color checkpoint or autonomous acceptance exists. [D2 report](../evidence/dynamic-A/rim-d2.json).

Two no-fit alternatives also fail on the same development frames: retained fine/affine correction with repaired coarse crops reaches 48.54/33.15 mm train/validation maxima; fixed projected rim-edge alignment reaches 27.05/24.27 mm. Neither consumes reserved images or activates a model. [Reuse diagnostic](../evidence/dynamic-A/rim-reuse-probe.json), [edge diagnostic](../evidence/dynamic-A/rim-edge-probe.json).

An exploratory replay blends the old and d2 fine predictions before initial temporal confirmation. Equal weights confirm all 207 episodes, with train/validation maximum confirmation errors 2.177/2.513 mm. However, raw initial validation frames reach 3.041 mm, and no stationary-cup tracker or explicit disappearance/movement rejection has been implemented. This is a development hypothesis, **not a passing localization candidate**; thresholds remain unchanged. The reproducible CPU helper is `probe_dynamic_rim_temporal.py`. [Replay](../evidence/dynamic-A/rim-temporal-feasibility.json).

The development pass closes with all 63 calibration scenes audited/reviewed, all 18 reserved moving scenes unused, no successor activation, and no component job running. All 2,284 protected parent files and 135 runtime sources verify unchanged against the current manifest. [Closure and runtime witnesses](../evidence/dynamic-A/development-closure.json).

## Operation and artifacts

Native root: `/workspace/learning/A-scale-dynamic-20261109-v1`; Ubuntu host root: `/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1`. Small reports are copied to `evidence/dynamic-A/`; weights, camera sheets, seeds and HDF5 recordings stay outside Git. Unchanged baseline plan/audit/locator files are copied natively from the completed parent study and are already versioned under `evidence/scale-A/`.

`watch_dynamic_stage.py` runs a finite, once-only calibration queue. `check_dynamic_stage.py` audits records and measured seeds. `finalize_dynamic_protocol.py` requires actual camera review before binding high seeds to moving scenes. `dynamic_level_data.py` admits reviewed whole episodes and checks localization before fitting; moving labels are matched to each frame's timestamp. `fit_dynamic_pair.py` trains independent RGB/color regressors and retains unsuccessful candidates. `pour_dynamic_study.py` requires frozen development acceptance before reserved evaluation, and separate explicit all-gate acceptance before autonomy.

The owned container keeps 4070-only GPU access, a four-core CPU quota on CPUs 16–23, shares 256, and 16 GiB memory/swap. Its priority guard remains active. Only the idle owned simulator was restarted to load eight calibration support files. No foreign process was signaled or reconfigured; snapshots do not prove uninterrupted operation of other workloads.


### Current continuation commands

Run native CPU helpers through `/isaac-sim/python.sh` with `CUDA_VISIBLE_DEVICES=` inside the owned container. Host observers and the deployment/component wrappers run with host `python3`. Keep collection and GPU fitting sequential.

1. Collection and actual visual review are complete. Keep all original recordings, failed candidates and frozen-input archives. Do not rerun consumed IDs or overwrite exclusive reports.
2. Coarse d2 passes internal prerequisites; fine d1/d2, detector reuse and edge alignment fail. No paired r4 fit is authorized by the development gates yet.
3. Reproduce the exploratory CPU replay with `/isaac-sim/python.sh /workspace/learning/A-scale-dynamic-20261109-v1/probe_dynamic_rim_temporal.py`, using `CUDA_VISIBLE_DEVICES=` and nice 10 inside the owned container. It asserts exact equality with the saved diagnostic and accepts nothing.
4. Implement a separate stationary-cup tracker candidate only with explicit disappearance/movement rejection and the retained accuracy requirements. Re-evaluate all old/new development scenes before selecting any successor detector. The fusion replay alone is insufficient.
5. Only after a fully passing detector exists, freeze its exact selection in a new paired-fit protocol and run the independent RGB/color fit on the guarded 4070. Do not point `prepare_dynamic_fit.py` at failed rim d2.
6. Any development failure keeps all 18 reserved cases and autonomy unused. Frozen component selection permits reserved perception evaluation only. Autonomous pouring additionally requires runtime integration/parity, all-size motion clearance and explicit all-gate acceptance.
