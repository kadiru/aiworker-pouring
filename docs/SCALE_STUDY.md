# Cup and bottle size continuation

Started November 9, 2026, after the user accepted two lost particles and provisionally allowed up to five per task. The completed parent stays frozen at **28/30 strict zero-loss passes**, or **30/30 under the later practical tolerance**. This is a separate study, `/workspace/learning/A-scale-20261109-v1`, prefix `SC_A_`. No resized autonomous pour has run and the parent robot checkpoint remains active.

Latest result: the 54 additional recordings and r3 refit are complete. The best validation maximum is **5.6184 mm**, above the 3 mm gate. No level fit or reserved test ran; all nine remain unused. See the completed continuation below.

## Scope and acceptance

The cup family crosses usable-height and diameter scales **0.9, 1.0, 1.1** independently. Base/wall thickness, original bottle, Material A, 1.8 mm particle resolution, cohesion 0.01 and 32 solver iterations remain fixed. Dimensions are declared fixture inputs, not visually estimated sizes. Particle trajectories and object transforms provide offline labels and scores only.

The unchanged perception gates are localization maximum **3 mm** in both whole-episode splits; per-size level MAE **2 percentage points**, near-full maximum absolute error **3.5 points**, maximum overestimate **2.5 points**, and CPU/GPU parity. Synthetic absent-cup masks provide a confidence diagnostic; they do not replace real missing-cup validation. Selection must pass every declared size. Static acceptance alone cannot enable autonomous pouring.

Future task reports retain both zero loss and provisionally **at most five lost particles**, with the existing grasp, stop, headroom, side-return and placement requirements. A changed particle resolution requires reconsidering that count-based tolerance. The method remains supervised visual perception and motion imitation with programmed control, **not RL**.

## Completed physical seeds and videos

The first dense-grid seed failed: the completed smallest cup ejected **1,990 particles / 11.60568 nominal mL** and settled at 76.53% rim height. A second record was interrupted and seven were unstarted. The complete and partial records remain preserved, without retrying their IDs.

A replacement starts from a physically recorded low layer without stretching particles, then uses an authored **8 nominal mL/s** jet beginning at 0.5 s. It stops after two measured samples at or above 90% rim height or a 15-second feed limit, then settles three seconds. Inventory checks prevent deletions or delayed clock catch-up injections. This is privileged offline calibration with a parked robot, not learned pouring. The historical source task had spill; only its retained cup particles were selected and independently revalidated.

All nine replacements pass recording, zero-loss and final-second stability checks (21 final samples each). Median settled levels span **90.3845–91.1275%**. All nine final four-camera composites and 36 selected encoded frames were inspected.

- [Three bounding shapes](../evidence/scale-A/pilot-r1.mp4): 20.85 s at labeled 2×, with one-second final holds; full decode and 12 native-source comparisons pass (maximum mean error 0.6682/255).
- [Six remaining shapes](../evidence/scale-A/pilot-r2.mp4): 38.2 s at labeled 2×; full decode and 24 source comparisons pass.

These are complete calibration clips with selected-frame manual review, not continuous manual playback or autonomous demonstrations. Native seeds, HDF5 and video artifacts stay outside Git.

Before fitting, the frozen parent confirmed **8/9** seed cups with worst localization **5.6327 mm**. Bounding-shape level MAE was 2.2861 points; the other five confirmed cups had 1.6645 points. These are consumed development observations, not independent generalization evidence.

## First completed static collection

Plan v4 SHA `530577c45edff0872ca23091a4803cf1a64ffa18559b649c212e7f5ee19c9415` declares **36 training + 18 validation episodes**, four/two per size. Four training and two validation layouts come from established normal/swapped families. Training cuts 0.22/0.50/0.78/1.0 rotate by size; validation cuts 0.30/1.0 alternate by size. A seed cut is not a physical surface label.

All **54/54** stationary 2.5-second recordings pass native four-camera integrity, zero-loss and final-second stable-label checks. Actual measured levels span **19.7273–91.1543%**. Nine contact sheets cover all 54 final cup-detail panels; every panel was actually inspected. The other three cameras pass recording checks but were not all manually viewed in this collection. The detail camera records at 1024 pixels and all cameras at 20 Hz. These are authored perception scenes, not robot demonstrations.

Nine near-full static holdouts, one per size with fixed small position jitters, remain **unstarted**. All 63 collection/holdout CPU geometry screens pass. The observer queues only unstarted IDs and handles the runtime queue cap without rerunning completed scenes. [Final audit](../evidence/scale-A/collect-r0-audit.json), [bound camera review](../evidence/scale-A/collect-r0-manual-review.json).

## Supervised locator fitting

The CNN has **42,947 trainable parameters** and predicts a rim-center heatmap with subpixel offsets. Training minimizes focal heatmap loss plus **2 × smooth-L1 offset loss**, with AdamW, 240 epochs / 7,200 updates per fit. Each size has 1/9 sampling probability; episodes within each size receive equal probability. There are **468 training images / 234 validation images** from the 36/18 whole-episode split (12 early frames plus one final frame per recording). No holdout images enter fitting.

| Round | Change | Best validation maximum | Result |
|---|---|---:|---|
| r0 | Parent initialization, integer translation, 384-pixel inputs | 5.9966 mm, epoch 50 | Rejected; 149.15 s fitting |
| r1 | Continuous bilinear translation, initialized from rejected r0 | 6.1170 mm, epoch 20 | Rejected; 165.62 s fitting |
| r2 | 768-pixel inputs from the same native images, initialized from rejected r1 | 12.2256 mm, epoch 40 | Rejected; 515.74 s fitting |

Low-filled validation cups dominate r0 errors. Two actual native diagnostic images show vertical rim-center offsets and adjacent four-pixel heatmap-cell mistakes. These observations motivated r1; its failure does not establish pixel phase as the cause. Two synthetic CPU checks verify translation/target agreement and identity pixels. R2 changed the input scale, scales camera intrinsics, augmentation, absent masks and temporal pixel radius consistently, and preserves the 3 mm world-space gate. Its unchanged 129-pixel receptive field covers half the previous scene extent. CPU projection round trips and 384/768 network shapes pass; these checks make no learned-accuracy claim.

Rejected reports, checkpoints, source versions and initialization wrappers are preserved under native `locator-rounds/r0` and `locator-rounds/r1`; original logs remain in `locator-fit-r0` and `locator-fit-r1`. The next-round declarations precede fitting. The active `best.pt` is never overwritten. [r0 report](../evidence/scale-A/locator-r0.json), [r1 report](../evidence/scale-A/locator-r1.json), [r2 declaration](../evidence/scale-A/locator-refit-r2-plan.json), [r2 measured result](../evidence/scale-A/locator-r2.json).

The coverage audit finds **no paired low/near-full observations at the same training position within any size**. Each of four positions carries only one fill per size. A prepared supplement adds the missing **54 training scenes**, producing 90 training / 18 unchanged validation episodes. It retains the nine unused holdout identities and adds no positions. Native CPU seed selection computes the initial inventories, but the proposal is **not queued or admitted**: it still requires a new plan, geometry/isolation checks, recording/stable-label audits and actual camera review. This is a targeted next data experiment, not an established cause or a promised accuracy improvement. [Coverage](../evidence/scale-A/train-fill-layout-coverage.json), [proposal](../evidence/scale-A/fill-pair-supplement-proposal.json).

The r2 final epoch has 247.6080 mm worst validation error and a negative synthetic-absence margin despite much smaller training errors. Higher resolution did not solve generalization. All three fits are closed, no accepted locator/level artifact exists, and all nine holdouts remain unstarted. The active checkpoint, runtime sources and protected parent files are verified unchanged. The final queue is empty and recording inactive; the 4070 guard/caps remain intact. [Verified closure](../evidence/scale-A/static-fit-closure.json).

## Subsequent gates, not yet executed

Only after the locator passes all-size per-frame and actual temporal confirmation will level fitting run. It freezes the retained CNN and fits a separate **139-input, 32/16-tanh residual readout**, 5,025 new parameters, for 1,200 AdamW steps. The objective is equal-size/episode weighted squared height error plus **0.02 × squared correction**. Crops use the image-localized cup and declared dimensions; no simulator position is used to center inference crops. The retained independent color estimator is diagnosed separately, not silently accepted.

Only after both components pass are their artifacts and evaluator hash frozen. The nine unused static scenes are then scored once on CPU, using a fixed 12-frame localization budget. The frozen parent is evaluated on the same scenes as a predeclared reference, never selected as an alternative from holdout outcomes. Selected and parent models use their respective calibrated input resolutions. No retry or fitting is allowed from this evaluation.

Current static data stop around 91%. **At least 94% coverage, held-bottle/occluded and dynamic-stop perception, independent color agreement, all-size approach/pour/return clearance and fresh autonomous tests remain required.** Low-fill independent holdouts are also not established by the nine near-full cases. No unknown-size perception or physical-robot claim is made.

## Bottle-size preparation: computational only

Proposed bottle height/body-diameter scales are **0.95/1/1.05**, with the mouth fixed. Runtime bottle scaling remains disabled.

- All nine one-layout upright envelope screens show a clear open hand/bench and an opposed fingertip contact bracket at 35–37.5° closure. Six original LP unknown statuses are retained; independent face-normal/edge-cross-product separating-axis checks pass. This is not force closure or physical pickup.
- Integrating the 20-sided visual shell and circular base, normalized to the authored 180 g baseline, proposes **164.03–196.70 g**. Effective density 3,548.07 kg/m³ is a modeling normalization, not measured glass density. Native collider-based mass/inertia and dynamic grasp remain unverified. Three analytic checks pass.
- The unchanged sparse body-grid initialization admits **85.747896–122.629464 nominal mL** across variants. **Four of nine cannot fit the former 100 nominal mL grid.** These are grid limits, not physical bottle capacities; feasible initial fills and full/empty targets must be declared accordingly.
- Inverting a perfect neck pixel at the old 135 mm plane would introduce up to **23.0262 mm XY bias** across six layouts. Lip height changes ±6.55 mm and grasp height ±2.2 mm. Correct-height inverse projection reconstructs within 1e-10 m; recorded and retained source-camera matrices match. These are analytic predictions, not observed detector errors.

Geometry, masks, source localization/tracking, grasp/pour frames, mass/inertia, initialization and all motion clearances must be consistent before bottle variation is enabled. Reports: `bottle-grasp-envelope*.json`, `bottle-mass-proposal.json`, `bottle-initial-grid-proposal.json`, `bottle-fixed-frame-proposal.json`.

## Execution and continuity

Use only `aiworker_pouring_4070`, immutable ID `0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`, the existing 4070 assignment, four-core quota on CPUs 16–23, shares 256, 16 GiB memory/swap and priority guard. Preserve the parent's source/checkpoint/reports by hashes. No foreign job is signaled or reconfigured. A priority stop must not trigger an automatic restart.

The shared host Docker daemon restarted November 9 at 09:00 PST, interrupting the rejected dense-grid batch. Initiator is unknown; no container OOM or priority-guard stop was found. Only the owned container was recovered after idle-GPU/cap checks, with `FOUNTAIN_RESUME=1` and its guard restored. This task did not modify Docker service or foreign processes.

Helpers have exclusive output paths. **Do not blindly rerun completed rounds.** Archive rejected outputs before a separately declared refinement. The completed r2 host invocation was (do not rerun its exclusive output path):

```sh
python3 /home/kadir/aiworker-pouring-4070/learning/A-scale-20261109-v1/run_scale_locator_fit.py \
  --component locator --round r2 --resolution 768 \
  --refine-from locator-rounds/r1/refinement-init.pt
```

After an accepted locator, the guarded host level command is `run_scale_locator_fit.py --component level`. After accepted level fitting, use `select_scale_static_components.py` inside the owned container with CUDA disabled. Only then may the host observer run `/tmp/aiworker-scale-stage-transfer/watch_scale_stage.py --part holdout --round r0`; its CPU audit, camera review and frozen evaluator must follow. No helper automatically activates a controller.

The first-stage next action was the paired-fill supplement; it has now been executed as documented below. A rejected static fit is not an autonomous task result.

## Completed paired-fill continuation

Plan v5 SHA `e8a89896054825ac3a5e3808bcde68836b016a92401f07d6048c611fed4f54d4` added exactly 54 new training IDs, retaining all validation/holdout specifications and gates. The old plan, audits, reviews, source versions and rejected r2 model were archived before mutation. All 108 collection and nine reserved-fixture geometry checks pass; ten local plan/admission checks pass. The observer ran only `collection_rounds.r1` once, followed by the complete 108-episode audit.

All **54 new recordings** are valid, have accepted stable labels and zero measured loss. All **108 combined episodes** are valid and accepted: **90 train / 18 validation**. New measured levels range **19.6846–91.1609%**. All 54 new final cup-detail views were actually inspected across nine sheets; the 54 older reviews carry forward through unchanged recording hashes. Other camera streams pass native integrity checks but were not all manually viewed. This is selected-frame review, not continuous playback. [Audit](../evidence/scale-A/collect-r1-audit.json), [review](../evidence/scale-A/collect-r1-manual-review.json).

The r3 locator fit used the retained single-detector parent, original 384-pixel integer-translation setup, 240 epochs and **17,760 AdamW updates**. There are 1,170/234 training/validation frames from 90/18 episodes. The parent artifact is a single detector, so the report correctly records `member: null`; the protocol's `member_0` shorthand did not select an ensemble member. All 42,947 parameters were fitted for **380.59 seconds** on the guarded 4070. No active robot weights were replaced.

The best attempt is epoch 240: **5.6184 mm maximum validation error**, **5.1479 mm training maximum**, **1.8124 mm validation mean**, and 0.63219 joint synthetic-absence margin. It fails the unchanged 3 mm limit. The rejected checkpoint SHA is `9be92cd1515aecf53e3793cf5dbf0d83f1ae4ff9a3a8f673d403650679397ab4`; it remains native as `cup-neural-localization-attempt.pt`. [Fit report](../evidence/scale-A/locator-r3.json).

The validation comparison verifies identical native recordings and all 234 frame identities. Worst error improves from 5.9966 to 5.6184 mm and overall mean from 1.83197 to 1.81241 mm. Low-fill mean improves from 2.84009 to 2.07182 mm, but near-full mean worsens from 0.82385 to 1.55301 mm and near-full maximum from 2.48327 to **4.57253 mm**. Four validation episodes exceed 3 mm. These are consumed validation measurements, not reserved-test performance; both training data and update count increased, so a causal claim about pairing alone is unwarranted. [Full comparison](../evidence/scale-A/paired-fit-validation-comparison.json).

The conditional runner exited normally at **`localization_gate_failed`**. It ran no level fit, no component selection and **none of the nine reserved static scenes**. The active parent checkpoint, runtime sources and protected historical evidence are verified unchanged. Final queue is empty, recording inactive, and the 4070 guard/resource caps remain intact. No foreign job was signaled or reconfigured. No new controller was activated. [Pipeline closure](../evidence/scale-A/pair-pipeline-r0-completion.json), [stage summary](../evidence/scale-A/paired-stage-summary.json).

The completed commands were the host observer `watch_scale_stage.py --part collect --round r1`, native CPU audit `check_scale_stage.py --part collect --round r1`, local review `review_scale_frames.py --part collect --round r1 --plan plan-collection-r1.json --new-only`, and host `run_scale_pair_pipeline.py`. Exclusive outputs must not be blindly retried. Any next fitting experiment needs a separately declared revision and preservation of this rejected candidate.

Next: diagnose/refine rim-center localization using the existing full-resolution camera data. Added paired-fill coverage alone did not meet accuracy requirements. Keep the reserved test identities unused until both locator and level gates pass; higher-fill, held-bottle/dynamic perception and all-size motion/bottle-grasp checks remain outside this static result.
