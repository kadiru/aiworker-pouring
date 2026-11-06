# Twenty successful autonomous pours

The requested reel contains twenty distinct successful autonomous tests from the earlier W30 study, using the unchanged v18 controller. It is a **success-only selection from 23 successes in 30 reserved tests**, not twenty consecutive successes, a new test batch, or twenty successful wider/swapped layouts. The later fourteen-layout stress test remains **1/14**; its failures and original video are preserved.

The video is local at `evidence/workspace-A/twenty-successful-pours.mp4`: **308.1 seconds (5:08), 1280×720, 20 fps, 6,162 frames**, silent H.264/yuv420p. It includes task and method cards, two synchronized native camera views, numbered attempts, controller-phase labels, final outcomes and the benchmark context. Full episode time intervals play at labeled 2×, with 0.5-second initial and two-second final holds. No internal cuts, interpolation or generated simulation frames.

## Task and actual training method

AIWorker must locate the bottle and cup, grasp and lift the bottle, pour, stop when the cup is nearly full or the source is empty, then return upright, set down, release and retreat. The target is at least 90% final settled rim height or source residual at most `max(0.25 nominal mL, 1% initial)`, with sustained verified grip, liquid transfer, the correct stop and zero measured loss. This is simulation evidence; no physical-robot validation is implied.

The method is a hybrid of supervised learning and engineered feedback:

- **Pouring motion:** contextual behavioral cloning by ridge regression on joint targets; 35 training and eight validation demonstration episodes. These motion references are scripted simulator demonstrations.
- **Settled liquid level:** a 16-coefficient supervised ridge model fitted on 98 training / 28 validation episodes, with 3,412 / 960 frames and regularizer 0.001. Simulator measurements provide the supervision. These episode counts are not a claim of 126 human demonstrations.
- **Perception and control:** calibrated RGB templates, frozen neural visual models, inverse-kinematics visual grasping, grip verification, temporal empty detection, bounded top-ups and a programmed return/release sequence. This is not reinforcement learning or an end-to-end learned grasp-to-pour policy.

The level objective is weighted squared prediction error plus a ridge penalty, `sum_i w_i (prediction_i - target_i)^2 + theta^T Lambda theta`; non-intercept selected coefficients use lambda 0.001 and the bias penalty is 1e-9. This fitting loss is distinct from the whole-task success criterion. The runtime actor uses two RGB views, eleven measured joint positions/velocities, its own history and fixed calibration; simulator object/particle state is used for scoring, not as live actor input.

The checkpoint was previously fitted on the RTX 4070. Its SHA256 is `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b`. No model fitting, inference, simulator restart, Ubuntu connection or GPU operation was performed to make this reel. Local Mac rendering uses two ffmpeg encoder threads.

## Selection and measurements

All nine successful empty-stop cases are included. Eleven of the fourteen successful full-stop cases are selected for placement coverage: initialize with native W30 test 00, then repeatedly choose the most distant remaining case from the selected set in four normalized placement coordinates; ties follow ascending native ID. Full cases are sorted by native ID, nine are interleaved with the nine empty cases, and the remaining two full cases close the reel. The three omitted successes are native 04, 13 and 14. All seven failures remain in the original benchmark and earlier outcome reel.

The twenty selected episodes have small placement variations within the W30 domain: cup world XY ±24 mm around its calibrated center, bottle cup-frame XY ±27/12 mm. These are not the later full-table arrangements or swaps. All W30 reserved test/development names remain excluded from the supervised fitting manifest.

The eleven full cups finish at **90.521–93.908% of rim height**. All nine empty cases finish with **zero remaining source particles**. Every selected case completes return/release and passes the unchanged whole-task score.

“Zero measured loss” means no particles outside the vessels at the end and none sampled near the table during the episode. It does not mean no transient splash: selected cases have transient nonairborne outside estimates as high as five particles (0.02916 nominal mL) that return. Sampling is 20 Hz and physics is 400 Hz. Nominal mL is a particle-count conversion, not calibrated physical volume. Per-clip outcome labels show final measurements throughout the clip; they are not live policy estimates.

## Provenance and verification

[Selection and source checks](../evidence/workspace-A/twenty-successes-selection.json) record the native hashes, training metadata, selection rule and omitted successes. Twenty HDF5 recordings and eighty sidecars match the native integrity manifest. The twenty task scores are recomputed from saved metrics, decisions and samples; 9,151 transitions pass camera/action timing and continuity validation. Maximum camera/state timestamp skew is 6.729e-7 seconds.

[Video manifest](../evidence/workspace-A/twenty-successes-manifest.json) records the final MP4 hash, exact frame count, timeline and source frame identity for every episode frame. [Encoded-video review](../evidence/workspace-A/twenty-successes-review.json) records complete decoding, 83 inspected encoded frames (three cards plus initial/pour/stop-vicinity/final for every case) and 160 native-to-encoded camera crop comparisons. Actual review images are retained locally under `evidence/workspace-A/twenty-successes-review/`. The MP4, images, raw data and model weights remain outside Git; code and small evidence reports are versioned.

From the project root:

```sh
.venv/bin/python tools/render_twenty_successes.py --preview
.venv/bin/python tools/render_twenty_successes.py
.venv/bin/python tools/review_twenty_successes.py
```

The renderer intentionally refuses to overwrite an existing completed MP4; archive an existing export before reproducing it. Review reruns reset visual approval to pending, requiring inspection of the new encoded contact sheets.

The next engineering action remains wider-table camera/localization coverage and full-hand fixture clearance before collecting supervised motion examples for those layouts. This video makes the existing successful behavior reviewable; it does not repair the wider-layout failures.
