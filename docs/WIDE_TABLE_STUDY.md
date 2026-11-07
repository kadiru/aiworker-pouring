# Wider and swapped table placements

Status: all thirty once-only reserved tests are closed and independently scored: **18/30 strict passes**, **30/30 verified grasps/transfers/returns**, and **26/30 zero-loss attempts**. November 6, 2026, America/Los_Angeles. Native integrity and local byte/SHA verification pass all 99 recordings / 359 sidecars / 52,339 transitions, with no partials. All three videos are rendered, fully decoded and reviewed at selected encoded frames; delivery is complete.

## Frozen evaluation results

| Criterion | Result |
| --- | ---: |
| Verified physical and visual grasp | 30/30 |
| Liquid transfer and ordered return/release/retreat | 30/30 |
| Complete strict task | 18/30 |
| Near-full tasks passing all criteria | 5/15 |
| Empty-bottle tasks passing all criteria | 13/15 |
| Near-full cups actually at or above 90% final rim height | 9/15 |
| Empty-target bottles with zero remaining particles | 15/15 |
| Zero measured loss | 26/30 |

Four attempts lost seven particles total (0.040824 nominal mL); zero spill across all thirty was **not** achieved. Six full-target attempts ended below 90%; three reached at least 90% but stopped `fill_unconfirmed`. One underfilled attempt also spilled, so the failure categories overlap. Localization/pickup worked across these tests; the settled-level predictor and full-stop confirmation remain the main task limitations. The larger placement range differs from previous benchmarks, so this is not a like-for-like comparison of success rates.

Descriptive position/goal breakdown: normal/full **5/8**, normal/empty **6/7**, swapped/full **0/7**, swapped/empty **7/8**. The zero strict passes for swapped full-cup tasks are a material remaining limitation despite successful pickup in every layout. All thirty accepted localizations were measured offline: maximum bottle XY error **3.569 mm** against the nearest native pose, maximum cup XY error **1.337 mm** against the authored fixture center. These are test-set diagnostics, not a guarantee outside this placement plan.

The retained stop traces expose prediction error: video test 19 had a six-frame lower-quartile level prediction of **97.58%** at its full stop versus **85.24%** final measured height. The three unconfirmed full stops reached the ten-top-up cap; their lower-quartile predictions remained below the fixed 92.274% confirmation target. This report does not change that target after seeing test results. All four table-loss events were recorded during pour/settle phases before return, and none of the thirty sampled retreat/hold traces contains gripper recontact. Some final surface-height traces drop while cup particle inventory stays constant; the physical cause is not established here. See `evidence/wide-A/test-diagnostics.json` for timestamps, localization estimates and the final one-second traces.

| Video test number | Native ID suffix | Failure | Final rim height | Lost particles |
| --- | --- | --- | ---: | ---: |
| 04 | `test_03` | Spill; bottle emptied correctly | 48.12% | 2 |
| 06 | `test_05` | Full stop unconfirmed | 90.34% | 0 |
| 07 | `test_06` | Underfill and spill | 85.88% | 1 |
| 10 | `test_09` | Spill; full stop confirmed | 92.06% | 2 |
| 13 | `test_12` | Spill; bottle emptied correctly | 59.05% | 2 |
| 14 | `test_13` | Underfill | 88.82% | 0 |
| 18 | `test_17` | Full stop unconfirmed | 92.91% | 0 |
| 19 | `test_18` | Underfill | 85.24% | 0 |
| 22 | `test_21` | Underfill | 86.93% | 0 |
| 26 | `test_25` | Full stop unconfirmed | 92.17% | 0 |
| 27 | `test_26` | Underfill | 89.29% | 0 |
| 30 | `test_29` | Underfill | 88.48% | 0 |

Every native ID begins `XW30_A_`. [Independently verified results](../evidence/wide-A/final-benchmark.json), [all thirty CSV rows](../evidence/wide-A/final-benchmark.csv), [recording audit](../evidence/wide-A/recording-integrity.json), [backup verification](../evidence/wide-A/final-backup-verification.json) and [resource comparison](../evidence/wide-A/runtime-comparison.json) retain the evidence. No attempt was retried, excluded, or used to change the frozen model or score. The final snapshot preserves the initial 4070-only allocation, four-core CPU quota, CPUs 16–23, low shares, 16 GiB memory/swap cap and identity-bound GPU guard. No other job was signaled or reconfigured; snapshots do not establish uninterrupted foreign-job performance.

## Selected model and training

Final combined fit r5 took **288.03 seconds on the RTX 4070**. Runtime source commit: `07cd41c321c7a00229618bc0db912a38be388001`. Checkpoint SHA256: `4c8ab7bab4b5121004e47ac1458f6df8ead0bd481c381efa807c7be8195d3430`. The frozen scene-plan SHA256 is `1d2ff7b863c0dc0724b8cb5cc34dadfa18b6dccf8e38c1e29b2015dc67e84076`.

- Localization uses clipped RGB gradients, top-three bottle-template aggregation, a training-derived tabletop region, calibrated cup templates and temporal consensus. Twenty-four training references supply templates; eight validation references plus 24 explicitly consumed development recordings calibrate confidence/position handling. Held-neck tracking checks the lifted bottle.
- Settled level uses **340 ridge coefficients**, selected lambda **0.001**, fitted on **146 training / 36 validation episodes**, **4,519 / 1,128 images**. The input features include per-row red/color-edge profiles and spatial interactions. Episode/domain weights are parent/reference/consumed development **1/1/64**, with extra near-full weighting. Objective: `sum_i w_i (phi(x_i)^T beta - h_i)^2 + lambda ||beta||²`.
- Empty detection uses a **48,001-parameter wrist-crop CNN**, balanced binary cross entropy and AdamW: **2,159 / 738 train/validation crops**, 100 epochs and 1,700 optimizer steps; epoch 46 selected. The final eligible validation probe detects 51/72 empty frames across all four empty episodes, with zero eligible false positives. These are consumed validation results, not fresh-test accuracy.
- Motion is contextual joint-target behavioral cloning from **43 training / 12 validation scripted demonstrations**, with a programmed IK projection. The objective fits demonstrated joint targets by squared error. Grasp verification, staged top-ups, irreversible stop and upright placement/release/retreat remain programmed feedback logic. There is **no reinforcement learning** and no physical-robot validation.

The 24 consumed development recordings are r0/r1/r2 repeats across eight layouts, not 24 independent layouts. Development r3 is used for candidate selection: **6/8 strict passes**, all eight complete grasps/returns with zero measured loss. Two full-stop failures remain explicit. A documented selection option permitted evaluating this candidate with those known limitations; it did not relax the reserved test criteria. No reserved test entered training, threshold selection or candidate selection.

The final fit's near-full validation MAE is 1.033 percentage points, wider-view near-full maximum error 3.177 points, and consumed-development near-full maximum error 1.774 points. These favorable consumed-data estimates did not establish reliable full-stop generalization: the fresh benchmark above is the relevant task result. The learned full-stop target is 92.274%, while the independently measured task requirement stays 90%.

## Audit and video production

The finite delivery pipeline waits until all thirty recordings are closed, creates the native report, audits every study recording and sidecar, verifies local backup bytes/SHA256, and independently recomputes all thirty primary scores. It checks the frozen scene plan, committed runtime files, model identity and before/after resource snapshots. It cannot queue or retry a test, fit a model, or restart the simulator.

The three local reels cover tests 01–10, 11–20 and 21–30 respectively, including every failure. They show native overview, task and wrist cameras, the planned bottle/cup layout, final PASS/FAIL reasons and task/method cards. Complete attempts run at labeled 2× speed; the four spill cases run at 1×. There is no generated camera footage or hidden failed attempt. Recordings, weights and MP4s remain outside Git; small manifests and review reports retain their hashes and exact frame mappings.

| Local video | Duration | Final review |
| --- | ---: | --- |
| [Tests 01–10](../evidence/wide-A/thirty-wide-tests-part-1.mp4) | 247.05 s | [Part 1](../evidence/wide-A/video-part-1-review.json) |
| [Tests 11–20](../evidence/wide-A/thirty-wide-tests-part-2.mp4) | 212.45 s | [Part 2](../evidence/wide-A/video-part-2-review.json) |
| [Tests 21–30](../evidence/wide-A/thirty-wide-tests-part-3.mp4) | 206.60 s | [Part 3](../evidence/wide-A/video-part-3-review.json) |

The complete set is 666.10 seconds, 13,322 frames, H.264 1280×720 at 20 fps. Full decoding passes all three files. All thirty unique IDs and displayed outcome labels match the independently scored report. The 270 native/encoded camera comparisons have maximum mean absolute pixel error 0.979 on a 0–255 scale. Manual review covers 96 selected encoded frames: each opening/closing card and every attempt's initial, middle and final frame. Labels are readable, camera panels are populated and the inspected frames have no text overlap. This is not full continuous manual playback. [Final review summary and media hashes](../evidence/wide-A/review-summary.json).

Executed from the project environment:

```sh
.venv/bin/python -u tools/prepare_wide_test_delivery.py \
  --checkpoint 4c8ab7bab4b5121004e47ac1458f6df8ead0bd481c381efa807c7be8195d3430 \
  --snapshot-dir /tmp/aiworker-wide-selected-v22-r5
```

The pipeline's post-copy steps are `tools/summarize_wide_benchmark.py`, then `tools/render_wide_videos.py --part N` and `tools/review_wide_videos.py --part N` for N=1,2,3. `tools/diagnose_wide_tests.py` separately measures accepted localization error, first stop, sampled spill timing and post-release gripper contact without changing any score. Export/review tools preserve existing output and require fresh destinations for another export. Automatic review fully decodes each MP4, checks dimensions/FPS/counts, and compares 90 selected displayed camera views per reel against their native HDF5 frames. Separate manual inspection of the selected encoded frames is also complete.

Delivery resumed at local scoring after a reader raised `KeyError: 'error'` on the healthy final snapshot, which omits optional error keys. The reader now accepts absent optional errors while still rejecting populated errors, an active recording or a nonempty queue. Independent scoring subsequently passed. Only local scoring/rendering resumed; no recording, test, model or runtime was changed. The recovery receipt is `evidence/wide-A/delivery-recovery.json`.

## Task and evaluation

AIWorker's right arm must locate a bottle and cup from RGB, grasp and lift the free dynamic bottle, pour a red water-like drink, stop at at least 90% measured settled rim height or an empty source, return upright, place/release and retreat. Material A and native particle/vessel physics are retained. The robot receives task/wrist RGB, joint positions/velocities, its own history and fixed calibration. Simulator poses, contacts and particle counts supply training labels and independent scoring, never autonomous control inputs.

Native root: `/workspace/learning/A-wide-20261105-v1` in the existing `aiworker_pouring_4070` container, mounted from `/home/kadir/aiworker-pouring-4070`. Local raw backups: `data/material-A-wide/`; small committed reports: `evidence/wide-A/`.

The geometry-screened plan has 24 training and eight validation scripted references; eight separate development scenes; and 30 fresh reserved autonomous tests arranged in 15 exact swaps. Five spatial families include transverse/diagonal placements, with independent ±12 mm jitter and varied initial liquid levels. Each swap pair has one full-cup and one empty-source target. All layouts are distinct across splits. Tests are once-only after selection; every failure and partial attempt is retained. No teacher fallback, test retries or test-based refitting is permitted within this benchmark.

The new task lens is 30 mm (previously 40 mm), fixed 384×384 task/wrist/overview recording at 20 Hz, physics at 400 Hz. Overview is recording-only. Calibration is saved per recording. Exact vessel positions and fills are in the frozen plan, not inferred from video labels.

Primary pass requires verified physical and visual grasp, two-finger contact on at least 80% of sampled tipping frames, at least 1 nominal mL transferred, correct stop reason, completed upright return/release/retreat, no existing scoring gates, and the actual height/empty goal. Empty allowance is max(0.25 nominal mL, 1% initial source). Zero measured loss requires both final outside inventory (including airborne particles) and ever-near-worktop inventory to be zero. One particle is 0.005832 nominal mL. This sampled simulation criterion permits transient returning droplets and does not establish continuously splash-free or physical-robot performance.

## Development record (historical)

The following sections retain the diagnostic sequence and earlier candidates. Statements about pending work describe their state at that point; the selected model and current results are recorded above.

### Initial learning and engineered components

This is supervised imitation/regression with a programmed feedback controller, not reinforcement learning or end-to-end task learning.

- Localization: reference RGB gradient template banks, view-local support, confidence calibration and temporal consensus. A partial-reference CPU diagnostic found a 23 mm bottle error where robot/background edges contaminated the original neck patch. All twelve narrowed-neck candidates on these 16 training references stayed below 0.9 mm initial error; all four original-mask candidates failed. Eight production checks reproduce the diagnostic. Narrower masks enter the candidate search; all final choices still require the complete validation set. A separate lifted-bottle bank uses one held RGB template per training reference: the consumed-reference probe passed all 96 sampled visual predicates, with maximum 3.93 px neck error. Its final mask requires disjoint validation and five of six positive frames per reference with <=6 px error. Ground truth labels these offline templates/errors; controller inputs remain RGB and the expected lift projection. Positive tracking coverage is not a missing-bottle rejection test. Candidate settings are selected using reference/validation data; synthetic missing-object occlusions are explicitly synthetic checks.
- Settled height: frozen visual features and an episode-balanced ridge readout. Objective is weighted squared height error plus a quadratic coefficient penalty, `sum_i w_i (f(x_i)-h_i)^2 + lambda ||beta||²`. Near-full examples carry extra weight. New height labels use stable-inventory frames after transfer away from the cup; an upright bottle above it can occlude its surface. The stopping margin accounts separately for wider-view validation overestimation, so the larger historical dataset cannot hide a wide-view bias.
- Motion: contextual joint-target behavioral cloning from successful scripted pours. Context comprises seven measured lift joints, RGB-localized cup XY and an intercept. Targets are learned on a 166-angle grid by ridge least squares. A programmed IK projection enforces calibrated mouth geometry; the projected target is not claimed to be wholly learned.
- Empty recognition: a separate high-resolution wrist CNN is initialized from the parent wrist encoder and fine-tuned with episode/class-balanced binary cross entropy, AdamW, and training-only geometric/photometric augmentation. Its constant image crop comes from projected training bottle envelopes; no live bottle pose enters control. Source presence and the original visual networks remain unchanged. The earlier 65/193-parameter frozen-feature readouts failed validation, and their reports are retained. The first wrist CNN also failed due to false-positive frames. Diagnostic r8 selects epoch 44 with zero eligible false-positive frames, 51/72 eligible positive frames and all four empty validation episodes detected, retaining the existing probability thresholds. It fits 48,001 parameters on 2,159 training crops; 738 validation crops select the epoch. This paragraph describes component validation; the completed full fit is recorded below. Validation must reject false empty detections and cover every empty validation episode before deployment. Runtime still requires consecutive confidence, presence, tilt and low-flow evidence.
- Grasp/stop/return: visual lift consistency plus finger obstruction, settled-height feedback and irreversible stop/return states. Programmed ready/approach clearance and joint limits remain explicit.

Full fit r2 completed in 78.88 seconds on the 4070, checkpoint `a88578b0bf3d770bf986c1e900b739a17cb154f5b86d3c473b43966bf6467757`, committed runtime source `d461222`. Level: 122/36 training/validation episodes, 3,914/1,128 frames, 21 parameters. Motion: 43/12 demonstrations. Empty CNN: 24/8 episodes, 2,159/738 crops, 100 epochs/1,700 optimizer steps, selected epoch 53. All four empty validation episodes are covered, with 57/72 eligible positive frames and zero eligible false positives. Counts and hashes are in `evidence/wide-A/model-summary.json`; its fitted coefficient arrays remain outside Git. Autonomous development r0 finished 5/8; these fit metrics are not test successes. Frozen parent: v18 SHA256 `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b`.

The complete validation set falsified the initial narrow-mask approach: robot edges caused up to 318.67 mm source error. A repaired component search uses clipped gradients, top-three matching, and a calibrated tabletop rectangle learned from training positions with 25 mm padding. The selected matcher reaches 1.533 mm bottle / 1.266 mm cup maximum initial error and passes every temporal window. Held tracking retains the <=6 px neck-error gate; its follows radius is calibrated from training p99 grasp displacement plus 2 px, bounded at 18 px, because closure can shift a free bottle laterally. All 192 held reference predicates and eight unchanged-parent compatibility checks pass. These are consumed training/validation results, not fresh autonomous tests. Failed fit/component reports and exact sources remain preserved.

## Consumed development repair after r0

R0 produced five strict passes, one false source-loss spill, one localization rejection, and one level-overestimate underfill. All eight recordings remain retained and are explicitly consumed for candidate development. Only r0 initial views calibrate localization candidates/confidence; templates remain from 24 training references. Stable-inventory r0 views add level-regression labels, restricted to pause, placement, release, final hold and stationary localization-abort modes. Flowing and upright-over-cup occlusion frames are excluded. The original 36 validation episodes stay separate. No development trajectory supplies empty-CNN or motion imitation targets; reserved tests are untouched.

Candidate v20 adds raw/aligned presence fusion gated by measured finger obstruction, and slows early abort returns to at least 4.3 seconds on the existing reverse path. These are programmed safeguards, not learned motor behavior. The 4070 level component passes with 130/36 train/validation episodes, 4,068/1,128 frames, 21 coefficients, and 0.917 percentage-point overall validation MAE. Consumed development errors are separately reported. The combined r3 fit now passes with the same level metrics, 0.765 mm maximum cup calibration error, and no false-loss episode in 8,929 pre-stop held frames. Checkpoint: `238dec5548f5733aa8c3e3d5e1c70e2dcdac9bda6166ed76cf27bd2b9afd64de`. Complete closed-loop development remains required before freezing the 30-test candidate.

## V21 deployed after r1 failures

R1 cleared the original false source-loss and cup-localization failures, but case 03 stopped at 85.830% rim height and case 04 at 89.666%, both below the unchanged 90% goal. Case 03 also lost eight particles after placement: the open right finger recontacted the bottle during retreat, accelerating it to 0.840 m/s. The 0.5-second retreat requested up to 0.230 rad per frame against a 0.12-rad command limit; eight of ten retreat commands were clipped. This is recorded contact/trajectory evidence, not a conjectured fluid leak.

The local candidate retains measured paths but uses minimum placement/release/retreat durations 1.5/0.6/2.0 seconds. CPU replay reduces the maximum planned retreat increment across five closed cases to 0.06193 rad. Twenty-one controller checks pass; native contact regression remains pending. The level candidate offers 32/60-parameter feature sets with position interactions while retaining the original 21-column prefix; fifteen recorded-view checks preserve old checkpoint predictions exactly. Fitting will consume audited r0/r1 stable-level images (sixteen recordings across eight layouts), retaining reference-only empty-CNN/motion targets and original validation. Consumed near-full error must stay below two percentage points and contributes to the full-stop margin. Development labels allow eight particles of inventory tolerance (0.046656 nominal mL) so a few returning droplets do not discard an otherwise stable pause; reference/validation sampling retains its half-particle rule.

## Pilot failures and clearance correction

Initial four reference pilots are all preserved. 00 reached 93.114% rim height with zero measured loss; swapped 01 emptied with zero measured loss. 02 contacted a finger at initialization and lost 27.999432 nominal mL. 03 grasped/poured but lost 0.268272 nominal mL. The original successful references are 00/01; later explicitly named repairs add accepted supervision without deleting originals.

The first preflight included all twelve arm/palm/finger meshes but incorrectly screened against the inner fluid profile. The corrected envelope adds actual 2.4 mm walls and a combined 2 mm contact margin. It identifies the two affected starts. The candidate uses a fixed ready TCP 10 cm higher and level overhead translation before descent. Geometry audits cover initial vessel contact and 41 approach samples; they exclude native tracking dynamics and general robot self-collision. The raised-ready approach passed all 70 geometry checks. Repaired 02 passed; repaired 03 still lost 28 particles during withdrawal. An over-cup upright return and drain pause then produced a zero-loss 03 repair at 93.403% rim height, permitting general collection. Reference 04 subsequently lost one particle; it is retained as a failure.

Repair identities `T30_A_demo_train_02_ready_r1`, `03_ready_r1` and `03_return_r2` map to the original planned layouts. Repaired records do not erase failed originals. Training acceptance checks require closed recordings, verified grasp, zero measured loss and no scoring gates.

New scripted empty-source references hold the verified emptied bottle at its current tilt for 1.2 seconds before returning, producing stable empty-state images for supervision. The hold never applies to full-cup stops or autonomous evaluation; both object motion and liquid remain native physics.

A further candidate retains the early lift boost until 45 degrees, adds 4 mm lip clearance and moves the final upright drip position to the cup center. Native swapped-pilot logs showed real cup-wall contact during early tilt. Candidate piecewise vessel envelopes pass all 70 sampled paths; dynamic validation is pending. This candidate was deployed only after the original collection became idle. Both wet repairs passed with zero measured loss at 93.852% and 94.160% rim height; autonomous evaluation remains pending. Late one-particle losses in references 04/12 originate from the bottle during upright transfer (cup inventory unchanged). The refined candidate lengthens the over-cup drain pause from 0.4 to 1.2 s and return translation from 1.0 to 2.0 s. Its geometry audit and two wet reference repairs are separate from autonomous performance. Geometry constraints and return timing are explicitly engineered components.

## Fresh identities and source freezing

The initial plan used the historical `T30_A_` prefix. Before any new development/test was run, an existing historical T30 benchmark was found. New development/tests use unused **`XW30_A_`** identities; the in-flight references retain `T30_A_demo_*` with protocol 17. The original plan, the 38-name mapping and corrected plan are preserved. Only identifiers changed; all numeric fixture values, splits and seeds are unchanged. Current plan SHA256: `1d2ff7b863c0dc0724b8cb5cc34dadfa18b6dccf8e38c1e29b2015dc67e84076`.

Before reserved tests, archive candidate weights/runtime source and record source commit, checkpoint SHA, plan SHA and hashes of every used runtime source in `final-selection.json`. Test queueing refuses changed files and skips any already-started test, including partial attempts. Source code and evidence reports are committed/pushed; recordings, weights and fitted coefficient arrays stay outside Git.

## Isolation and reproduction

Owned immutable container: `0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`; RTX 4070 UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`. Retain four CPU cores of quota on 16–23, shares 256, nice 10, 16 GiB RAM/no extra swap. The identity-bound guard yields by stopping only AIWorker if foreign 4070 compute appears. Other workloads, including the 4090 container, are read only. These limits do not imply literally zero shared-host overhead.

Use `tools/deploy_wide_study.py` on the Ubuntu host for an idle-only archived deployment. It verifies ownership, immutable ID, GPU exclusivity and caps; refuses changes after test selection; and restarts only the owned container. Native Python is `/isaac-sim/python.sh`.

Local protocol checks: `python3 -m unittest discover -s tests -p test_wide_protocol.py -v`. Initial preflight and corrected audits are `tools/preflight_wide_study.py` and `tools/audit_wide_clearance.py`. POST commands to `http://192.168.68.60:8212/control`: `wide_repair`, `wide_collect`, `wide_train`, `wide_development`, `wide_test`, only in the appropriate verified idle stage. Do not reload/stop an active recording to advance stages. Final report/video commands and measured outcomes will be added after the actual evaluation.

The weighted v21 component passes with 60 ridge coefficients (lambda 1); parent/wide-reference/consumed-development episode weights 1/4/16. Training uses 138 episodes/4,272 images; validation 36/1,128. Near-full validation MAE 1.042 percentage points; maximum consumed-development error 1.996 points. The 92.496% stop target includes the observed overestimation margin. Full deployment and closed-loop development remain pending.

Combined fit r4 completed in 241.35 seconds on the 4070, source `826f865`, checkpoint `e01f5ff12eb8c7c6964d2954c9c05bfa0aef3bb6ac55a94d1da598c659de51f5`. It reproduces the weighted component level metrics. Empty validation coverage is 58/72 eligible positives with zero eligible false positives, covering all four empty episodes. Development r2 is running; no reserved test has run.

## Local reference storage

Closed scripted-reference HDF5 copies may be stored losslessly as `episode.hdf5.zst` to leave room for fresh test videos. `tools/archive_wide_references.py` verifies original bytes against the native index, compresses with one CPU thread, decompresses and rechecks every byte/hash, then replaces only the local raw duplicate. Native Ubuntu originals remain unchanged. Backup and final-score verification stream-decompress these archives; explicitly reviewed development copies can also be archived with `--development-rounds`. Every selected development round must have eight manually inspected records with matching source hashes. Test recordings stay directly readable. To inspect a reference with an HDF5 reader, restore a temporary copy with `/opt/homebrew/bin/zstd -d -k path/to/episode.hdf5.zst`. The archive remains intact.

## V22 fine surface features

The 60-column fit and 144 reweighting variants failed after incorporating r2. Five-row pooling was replaced by additional per-row red-mask, continuous red-edge and spatial-interaction features, preserving the original 60-column prefix exactly on fifteen native crops. The component selects 340 coefficients, ridge 0.1, with parent/wide-reference/consumed-development weights 1/1/64. It uses 146 training / 36 validation episodes and 4,519 / 1,128 images, including 24 explicitly consumed r0/r1/r2 recordings. Near-full validation MAE is 1.040 percentage points; wider near-full maximum error is 2.881 points; consumed near-full maximum is 1.988 points. Stop target: 92.488%. These are calibration/validation estimates, not independent task success. All 32 existing controller/source/settled tests pass after supplying a missing test module; the initial import failure is retained locally. Eight protocol/scoring checks pass.

R2 native audit covers 61 recordings, 207 sidecars and 28,066 transitions, all locally byte/SHA verified. Sixty-four selected r2 native frames were inspected. Every r2 release/retreat trace has zero recorded finger recontact; strict outcomes are unchanged. Reviewed reference/development archives retain native originals and verify exact decompression hashes to conserve local disk space for the requested test videos.

The combined r5 fit recalibrates localization across 56 reference/development scenes before recropping level images. Its final metrics differ from the component diagnostic: near-full validation MAE 1.033 points, wider near-full maximum 3.177 points, consumed near-full maximum 1.774 points, target 92.274%. Checkpoint `4c8ab7bab4b5121004e47ac1458f6df8ead0bd481c381efa807c7be8195d3430`, source `07cd41c`. Presence replay covers 14,748 pre-stop held frames without a false-loss episode. The wrist CNN selects epoch 46. R3 autonomous development is running, not yet scored.

## Explicit evaluation with known full-stop limits

Before any reserved test, the selection helper now permits a documented opt-in to evaluate conservative full-stop failures (`underfilled`, `fill_unconfirmed`, `level_unconfirmed`, `wrong_stop_condition`). This changes the former requirement for eight development successes; it does **not** change any task score or test threshold. Every development case must still have verified/sustained grasp, transfer, complete ordered return/release, zero measured loss and no teacher fallback. The rationale, actual failure labels and helper hashes are retained in `final-selection.json`. The default still requires eight strict passes. Selection prioritizes zero measured loss and complete grasps/returns, then development strict passes; ties favor the latest fit. All thirty unused reserved scenes then run once with immutable source/weights/plan and honest video labels. This bounds further tuning while preserving the user-requested evaluation.

Thirteen protocol, scoring and selection-gate tests pass, including rejection of a recorded spill and retention of underfill/unconfirmed-stop failure labels. The active r3 fourth case has latched `fill_unconfirmed` at its top-up limit; its final result is pending. No test data have been used.

R3 is selected among r0–r3 using the stated zero-loss/grasp/return and strict-pass criteria (ties choose the latest fit). All four full cases finish above 90% rim height, but cases 03 and 07 remain failures because ten top-ups expire before confirmation. Their six-frame prediction quartiles are 90.263% and 92.094% against the unchanged 92.274% target; final measured levels are 91.038% and 96.333%. The selected model remains unchanged for the reserved evaluation. Native audit and manual frame review precede freezing.
