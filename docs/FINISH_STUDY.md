# Settled-level repair and natural bottle withdrawal

Authorized November 6, 2026 (America/Los_Angeles). **In progress.** The previous wider/swapped benchmark remains 18/30; it is not relabeled. No new autonomous development or reserved test has completed yet.

## Current implementation and evidence

The terminal motion now opts into zero stationary drain hold. It turns upright over the receiving cup, immediately follows a raised transfer to the original bottle location, then gently places, releases and retreats. Commanded return duration is 3.1 seconds instead of 4.3 seconds. The old checkpoint retains its 1.2-second hold by default. Video and measured motion must still verify this change during autonomous pours. A read-only audit of the historical thirty-test cohort measures 0.9–1.2 seconds of upright near-stationary dwell (median 1.1 seconds), with no post-release gripper recontact. The new development acceptance bound is ≤0.35 seconds under the recorded 3D-speed/tilt-speed definition, zero recontact, and a 3.1-second terminal return within one sample.

All 38 new approach layouts and 38 continuous-return paths pass geometry screening. Return screening covers 63 poses and articulated closed-finger hull ranges; this is not dynamic, self-collision or no-spill validation. Thirty CPU controller/feature/calibration checks pass. Protocol 18 now uses the raised home and wide-layout path support already present in protocol 17.

The first three valid stationary grid pilots reached 66.70%, 78.86% and 82.87% settled rim height with zero loss. Three new settled-seed pilots reach 84.28%, 90.00% and 94.10%, with zero command variation, zero tail arm motion, constant inventory and height standard deviation below 0.00030. Their three final task-camera frames are manually inspected. The remaining 45 calibration cases are queued; `/tmp/aiworker-finish-calibration-v2` is a finite audit → fit → development handoff. It stops on failed checks and cannot freeze or launch reserved tests.

## Task and evaluation

Locate, grasp and lift the bottle, pour red liquid into a material-A cup, stop at at least 90% settled rim height or when the source is empty, then turn upright, withdraw, place, release and retreat. Runtime inputs are task/wrist RGB, measured joints, own history and fixed robot/camera/bench calibration. Object transforms and particle state provide labels and scores only.

New namespace `FN30_A_`, protocol 18, native root `/workspace/learning/A-finish-20261106-v1`. Eight development scenes and thirty reserved scenes use seeds 2026110613 and 2026110614. Fifteen exact bottle/cup swap pairs, fifteen full and fifteen empty goals, five placement families and ±12 mm jitter. The 60-second maximum duration was declared before any new autonomous test, to accommodate slower pouring. The reserved plan SHA is `25d5ee5221ce280ad05bf793e3552fd0ddd00fe9cb6a6738e069d50d4e357e45`.

Strict success requires physical and visual grasp confirmation, sustained bilateral contact ≥80%, ≥1 nominal mL transfer, the correct stop reason, an ordered return/place/release/retreat, policy completion, no gates or fallback, and the full/empty goal. Full means measured rim height ≥0.90; empty means source inventory ≤max(0.25 nominal mL, 1% initial). Final outside particles, including airborne particles, and ever-near-worktop particles must both be zero. No-pour aborts and underfilled cups fail. Measurements are sampled at 20 Hz; nominal particle-count mL is not calibrated physical volume, and transient droplets that return are allowed by this score.

Freeze sources and weights after development review, then run all thirty once. No reserved retries, fitting, threshold changes or source changes during evaluation. Every failure remains in the report and videos.

## Learning method

This is a hybrid supervised system: RGB localization, a wrist empty-state CNN, contextual behavioral cloning plus IK, a cup-level regressor and programmed feedback/terminal logic. It is not reinforcement learning or an end-to-end learned manipulation policy. The localizer, empty CNN and cloned motion weights are retained from the parent checkpoint; this follow-up refits the liquid-level component on the RTX 4070.

The cup crop normalizes position, scale and shear from visual localization and calibrated rim/base geometry. Candidate features include the previous 340 coefficients, color-envelope quantiles and spatial interactions, up to 604 features. Ridge regression minimizes weighted squared height error plus an L2 coefficient penalty. A validation-selected nonlinear residual is also considered: 34 inputs, tanh layers of width 32 or 64 then 16, AdamW at 0.001 for 1,200 steps per width, evaluated every 20 steps. Its loss adds weighted squared error and a residual-size penalty. A candidate is retained only if the unchanged consumed-validation near-full maximum error is below four percentage points and the derived stop target is at most 0.945. Independent task success remains 0.90.

The completed wide benchmark's thirty test recordings and eight r3 development recordings are explicitly consumed. Former test swap pairs 3, 8 and 13 are validation; the other former tests and r3 are training, alongside the inherited cache. These validation outcomes are already known and are not unbiased new-performance evidence. Stable-inventory settling-pause and final views have per-frame native hashes, image indices and simulator labels. Initial FN30 development and reserved images are excluded from fitting.

Candidate engineering parameters are slow tilt 2.5°/s (previously 4), braking 45°/s (90), trim rate 18°/s (35), coarse flow cap 10 nominal mL/s (18), coarse brake target 0.87 (0.89), and trim timeout 2.5 seconds (1.5). These are changes to validate, not newly learned motor skills. Liquid physics and autonomous initialization are unchanged.

## Calibration and preserved failures

The current calibration plan retains three stable low-level pilots plus 48 stationary seed-based cases: three levels at twelve training and four validation layouts from previously consumed references. The robot remains at its initial pose throughout each 2.5-second recording. Final measured settled height labels images from 1.5 seconds onward. These are camera calibration, never grasp/pour successes.

The seed is the cup-contained particle arrangement from native historical trajectory `20261102T194804Z/076-robot_full-790609`. Its liquid settings and cup geometry match the current fixture. Particles are transformed into the cup frame, clipped at declared initial heights, translated to each calibration layout, initialized with zero velocity and allowed to settle natively. Source trajectory, parameters and derived seed hashes are recorded in `calibration-seed-provenance.json`. The historical source itself spilled; only its inside-cup arrangement is reused and no source success is implied.

Preserved failed development steps:

- Four historical-data fits (ridge, bias calibration, perspective features and nonlinear residual) failed the accuracy gate. `fit-attempts/r0`–`r3` retain their sources and diagnostics. None produced a new checkpoint.
- Calibration v0 in run `20261107T051405Z` unexpectedly entered the inherited timeout return at 0.1 seconds. Its first case moved the arm and lost 6.304392 nominal mL. Only the owned queue/handoff was stopped. Invalid/aborted records and the old plan are retained and excluded.
- Explicit stationary v1 fixed that timeout path. Its sparse grids stayed stable but only reached 66–83% fill.
- Three denser grid pilots lost 9.99–15.24 nominal mL while stationary. Their records and plan remain retained and excluded. The settled-seed initializer replaces the unrun dense cases; it does not change evaluation fluid parameters.

## Resource isolation and reproducibility

Only the immutable `aiworker_pouring_4070` container is used: ID `0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`, GPU UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, four-core quota, CPU set 16–23, shares 256, 16 GiB memory/swap. The existing guard yields to foreign 4070 compute jobs by stopping only this container. The separate 4090 job is not stopped or reconfigured. Snapshots do not establish zero shared-host overhead.

Native HDF5 recordings, coefficients, checkpoints and private assets remain outside Git. Small provenance/reports and source are committed to the independent private repository. Because Mac disk space is limited, new raw recordings remain on Ubuntu. Media tools fetch one temporary source at a time, verify bytes/SHA, compose or inspect it, then remove only that temporary download. Scalar metadata is backed up locally. This is not a full local raw-data backup.

The review pipeline audits all camera decodes/timestamps and inventory partitions, independently recomputes all thirty scalar outcomes, checks source/weight/resource identities, and renders three complete labeled reels. Development review includes eight task/wrist/overview frames and ten return-sequence frames per attempt. The withdrawal diagnostic measures upright, near-stationary dwell over the cup and post-release recontact. Selected-frame review is not continuous manual playback.

## Next actions

Finish calibration and audit; fit on the 4070; inspect eight autonomous development cases; freeze a reviewed candidate; run thirty fresh tests; independently score, render and inspect all videos; update results and push final evidence. No new success rate is available yet.
