# Settled-level repair and natural bottle withdrawal

Authorized November 6, 2026 (America/Los_Angeles). **All thirty fresh tests closed November 8 at 00:03 PST: 25/30 strict passes, 29/30 zero measured loss, 29/30 completed return/release sequences.** Full-target tasks pass 12/15; empty-target tasks pass 13/15. Successful full cups settle at 91.490–96.259% rim height; all thirteen successful empty-target bottles finish with zero source particles. The previous 18/30 wider/swapped benchmark remains separate and unchanged. No reserved trial was retried and no controller, threshold, physics or checkpoint changed during this evaluation.

The programmed 1.2-second upright drain hold is removed. The bottle turns upright, follows a raised transfer to its original location, is placed and released, and the hand withdraws. All 28 completed post-pour returns take 3.10–3.15 seconds for upright/transfer; the separate pre-pour tracking abort uses a 4.30-second recovery. Among the 29 observed returns, maximum sampled stationary upright dwell above the cup is 0.05 seconds and there is no recorded post-release gripper contact. Case 13 never stops or returns; its missing-motion measurements are null, not zero-dwell successes. Tilted settling pauses during filling remain.

Native integrity validates **62 complete development/test recordings, 48,398 transitions and 248 complete-record scalar sidecars**. One catalogued incomplete recorder abort and its scalar sidecar remain separate: 249 scalar files total are backed up and SHA-verified locally. All thirty primary scores are independently recomputed from those records. See [results](../evidence/finish-A/final-benchmark.json), [timing](../evidence/finish-A/final-outcome-timing.json), [return measurements](../evidence/finish-A/test-withdrawal.json), [recording audit](../evidence/finish-A/recording-integrity.json) and [stop-gate diagnosis](../evidence/finish-A/final-stop-gates-diagnostic.json).

| Failed case | Audited observation |
|---|---|
| `FN30_A_test_01` | Two particles, 0.011664 nominal mL, are recorded near the worktop at 25.055 s while pouring, 0.40 s before the stop command. Empty detection and return complete. |
| `FN30_A_test_13` | Source remains empty from 25.455 s through 60.005 s. The wrist empty score is 98.61–99.86%, but bottle presence stays 58.88–65.70%: below the 70% empty-stop requirement and above the 5% lost-source abort threshold. It never stops/returns. Two-finger contact is recorded in 42.05% of tipping samples. No measured loss. |
| `FN30_A_test_15` | Final actual settled height is 95.6051%, but the stop estimator's six-frame lower quartile is 91.6981%, below the 92.5414% confirmation threshold. Thirteen pauses exhaust the budget; `fill_unconfirmed` is retained. Return completes with zero loss. |
| `FN30_A_test_19` | Final actual height is 92.6322%. The stable learned estimate is 93.7726%, but it disagrees with the geometry estimate by 10.3375 percentage points, above the 10-point limit. `level_unconfirmed` is retained; return completes with zero loss. |
| `FN30_A_test_23` | Presence scores of 2.17–2.95% trigger an early tracking abort at 4.655 s, despite encoder obstruction and bilateral contact in all eight corresponding states. No meaningful liquid transfer occurs. There are zero samples above 20° tilt: the frozen score's zero sustained-grip fraction does **not** establish physical grip loss. Recovery/release completes with zero loss. |

These five outcomes remain failures. The zero-spill and reliable-stop goals are not fully achieved. All source identities used by the diagnostic match the complete recording/scalar audit. The missing-return case still exhibits a long empty-bottle hold before a stop decision, distinct from the removed programmed terminal hold.

All three video exports and selected-frame reviews are complete: **three full decodes, 270 native-camera comparisons, 96 visually inspected encoded frames/cards**, maximum mean pixel error 1.024/255. This is selected-frame inspection, not continuous manual playback. All thirty complete native attempts and every failure appear at labeled 1×/2× speed, with no internal cuts, interpolation or generated camera footage. [Review summary](../evidence/finish-A/review-summary.json).

| Local video (outside Git) | Duration |
|---|---:|
| [Tests 01–10](../evidence/finish-A/thirty-finish-tests-part-1.mp4) | 261.75 s |
| [Tests 11–20](../evidence/finish-A/thirty-finish-tests-part-2.mp4) | 258.50 s |
| [Tests 21–30](../evidence/finish-A/thirty-finish-tests-part-3.mp4) | 229.70 s |

Final resource verification retains the frozen checkpoint, all 42 runtime files, asset, container limits and guard. The separate 4090 PID 232122 is active at 74% in the final snapshot; host available RAM is 47,537 MiB. No foreign job was signaled or reconfigured. These snapshots do not establish zero shared-host overhead or uninterrupted foreign execution.

Next: repair source-presence handling and bounded uncertain-state recovery, then the level-estimate/agreement failures. Further tuning must use a separate fresh evaluation; the completed thirty outcomes remain unchanged.

Across all 24 completed r0–r2 development attempts, upright stationary dwell over the cup is at most 0.05 seconds, with no post-release gripper contact. The bottle turns upright, immediately clears the cup, returns to its original position, releases and leaves the hand withdrawn. Strict task scores are 7/8, 7/8 and 4/8 respectively; zero-loss counts are 8/8, 7/8 and 4/8. All failures remain retained. R2 loses nine particles across four attempts, totaling 0.052488 nominal mL; the return itself completes in all eight.

V24 refits only the settled-level predictor, using 238 training / 54 validation episodes and 7,116 / 1,476 images. The 4070 fit takes 33.1 seconds. Its 126-coefficient ridge model plus 139→32→16→1 tanh residual has 5,151 fitted parameters. Existing near-full validation MAE is 0.8997 percentage points, maximum absolute/overestimate 2.3414 points. The resulting full-stop threshold is 92.5414%. These component measurements are not fresh autonomous task results. Twenty-one of the 24 consumed development records supply eligible training views; the other three remain in the manifest with zero eligible images. Existing validation membership and image labels are retained.

Training-pause labels now predict expected settled height using a monotonic map fitted from static calibration; final/static measured labels remain unchanged. Runtime uses RGB, not particle inventory. The V24 motion also shifts the lip 8 mm toward cup center between 116° and 122° tilt to address two observed late-empty losses. All 38 authored-layout geometry checks pass; geometry alone is not fluid validation. The fresh dynamic results above retain the remaining two-particle loss. It does not directly change the 80° settling pose where another r2 loss occurred. Physics and the natural return timing remain unchanged.

Current checkpoint: `5068af6b18583afcb3eff231110c63896e0be898f389b9aad940f9e9405179d3`. Runtime contains 42 files from `76302c480dd0cbf16ca4621c8b90b470c13cf5cc`; fitting helper source is `1ec4405eff9880f64dbb9078afbda926348b7331`. Native run `20261107T222043Z` is closed; `/tmp/aiworker-finish-test/` retains its finite observer output. Only the owned container was restarted, with its GPU binding, four-core/16 GiB caps and priority guard preserved. The separate 4090 job is active in the deployment snapshot; no foreign job was signaled or reconfigured.

The remaining sections record the method and a condensed development lineage; the original reports and Git history retain intermediate preparation states. Current evidence is in `final-benchmark.json`, `final-outcome-timing.json`, `test-withdrawal.json`, `runtime-comparison.json`, the fit/label manifests and the reviewed development reports.

## Current implementation and evidence

The terminal motion now opts into zero stationary drain hold. It turns upright over the receiving cup, immediately follows a raised transfer to the original bottle location, then gently places, releases and retreats. Commanded return duration is 3.1 seconds instead of 4.3 seconds. The old checkpoint retains its 1.2-second hold by default. The completed autonomous scalar audit verifies the timing above; video review is tracked separately. A read-only audit of the historical thirty-test cohort measures 0.9–1.2 seconds of upright near-stationary dwell (median 1.1 seconds), with no post-release gripper recontact. The new development acceptance bound is ≤0.35 seconds under the recorded 3D-speed/tilt-speed definition, zero recontact, and a 3.1-second terminal return within one sample.

All 38 new approach layouts and 38 continuous-return paths pass geometry screening. Return screening covers 63 poses and articulated closed-finger hull ranges; this is not dynamic, self-collision or no-spill validation. Thirty-one CPU controller/feature/calibration checks pass. Protocol 18 now uses the raised home and wide-layout path support already present in protocol 17.

The first three valid stationary grid pilots reached 66.70%, 78.86% and 82.87% settled rim height with zero loss. Three new settled-seed pilots reach 84.28%, 90.00% and 94.10%, with zero command variation, zero tail arm motion, constant inventory and height standard deviation below 0.00030. Their three final task-camera frames are manually inspected. All 51 calibration cases are now closed and audited: 2,499 transitions, all stationary and stable, zero measured loss, maximum tail-height standard deviation 0.000407. The finite v4 handoff stopped after fit r4 failed its accuracy gate; no new checkpoint or development queue was produced.

## Task and evaluation

Locate, grasp and lift the bottle, pour red liquid into a material-A cup, stop at at least 90% settled rim height or when the source is empty, then turn upright, withdraw, place, release and retreat. Runtime inputs are task/wrist RGB, measured joints, own history and fixed robot/camera/bench calibration. Object transforms and particle state provide labels and scores only.

New namespace `FN30_A_`, protocol 18, native root `/workspace/learning/A-finish-20261106-v1`. Eight development scenes and thirty reserved scenes use seeds 2026110613 and 2026110614. Fifteen exact bottle/cup swap pairs, fifteen full and fifteen empty goals, five placement families and ±12 mm jitter. The 60-second maximum duration was declared before any new autonomous test, to accommodate slower pouring. The reserved plan SHA is `25d5ee5221ce280ad05bf793e3552fd0ddd00fe9cb6a6738e069d50d4e357e45`.

Strict success requires physical and visual grasp confirmation, sustained bilateral contact ≥80%, ≥1 nominal mL transfer, the correct stop reason, an ordered return/place/release/retreat, policy completion, no gates or fallback, and the full/empty goal. Full means measured rim height ≥0.90; empty means source inventory ≤max(0.25 nominal mL, 1% initial). Final outside particles, including airborne particles, and ever-near-worktop particles must both be zero. No-pour aborts and underfilled cups fail. Measurements are sampled at 20 Hz; nominal particle-count mL is not calibrated physical volume, and transient droplets that return are allowed by this score.

Freeze sources and weights after development review, then run all thirty once. No reserved retries, fitting, threshold changes or source changes during evaluation. Every failure remains in the report and videos.

## Learning method

This is a hybrid supervised system: RGB localization, a wrist empty-state CNN, contextual behavioral cloning plus IK, a cup-level regressor and programmed feedback/terminal logic. It is not reinforcement learning or an end-to-end learned manipulation policy. The localizer, empty CNN and cloned motion weights are retained from the parent checkpoint; this follow-up refits the liquid-level component on the RTX 4070.

The cup crop normalizes position, scale and shear from visual localization and calibrated rim/base geometry. Candidate features include the previous 340 coefficients, color-envelope quantiles and spatial interactions, up to 709 features. Ridge regression minimizes weighted squared height error plus an L2 coefficient penalty. A validation-selected nonlinear residual is also considered: 34 or 139 inputs, tanh layers of width 32 or 64 then 16, AdamW at 0.001 for 1,200 steps per width, evaluated every 20 steps. Its loss adds weighted squared error and a residual-size penalty. A candidate is retained only if the unchanged consumed-validation near-full maximum error is below four percentage points and the derived stop target is at most 0.945. Independent task success remains 0.90.

The completed wide benchmark's thirty test recordings and eight r3 development recordings are explicitly consumed. Former test swap pairs 3, 8 and 13 are validation; the other former tests and r3 are training, alongside the inherited cache. These validation outcomes are already known and are not unbiased new-performance evidence. Stable-inventory settling-pause and final views have per-frame native hashes, image indices and simulator labels. That initial V23 fit excluded all FN30 images. The selected V24 refit subsequently consumes 24 r0–r2 development records as explicit training data; the r3 acceptance round and all thirty fresh test images remain excluded.

Candidate engineering parameters are slow tilt 2.5°/s (previously 4), braking 45°/s (90), trim rate 18°/s (35), coarse flow cap 10 nominal mL/s (18), coarse brake target 0.87 (0.89), and trim timeout 2.5 seconds (1.5). These describe the initial V23 engineering changes, not newly learned motor skills. The selected V24 uses coarse brake target 0.82, maximum 13 pauses, full threshold 0.925414 and an 8 mm late lip shift. Liquid physics and autonomous initialization are unchanged.

## Calibration and preserved failures

The current calibration plan retains three stable low-level pilots plus 48 stationary seed-based cases: three levels at twelve training and four validation layouts from previously consumed references. The robot remains at its initial pose throughout each 2.5-second recording. Final measured settled height labels images from 1.5 seconds onward. These are camera calibration, never grasp/pour successes. A CPU diagnostic found initial localization on already-full calibration cups could select background edges, with a maximum 213.68 mm error among the first fifteen closed records. The fitter instead takes the initial RGB cup estimate from the matching accepted historical low-fill reference, as autonomous inference does before pouring. Known fixture positions select the corresponding supervision record only; simulator poses do not define runtime crops. Each anchor records its source hash, initial frame indices and measured localization error (<3 mm). The autonomous localizer is unchanged.

The seed is the cup-contained particle arrangement from native historical trajectory `20261102T194804Z/076-robot_full-790609`. Its liquid settings and cup geometry match the current fixture. Particles are transformed into the cup frame, clipped at declared initial heights, translated to each calibration layout, initialized with zero velocity and allowed to settle natively. Source trajectory, parameters and derived seed hashes are recorded in `calibration-seed-provenance.json`. The historical source itself spilled; only its inside-cup arrangement is reused and no source success is implied.

Preserved failed development steps:

- Four historical-data fits (ridge, bias calibration, perspective features and nonlinear residual) failed the accuracy gate. Fit r4 adds the 51 corrected calibration records but also fails: near-full maximum overprediction 4.2847 points and maximum underprediction 4.0481 points. `fit-attempts/r0`–`r4` retain sources and diagnostics. None produced a new checkpoint. R4 fit takes 18.11 seconds on the 4070 with 217/54 train/validation episodes and 6,247/1,476 images.
- Calibration v0 in run `20261107T051405Z` unexpectedly entered the inherited timeout return at 0.1 seconds. Its first case moved the arm and lost 6.304392 nominal mL. Only the owned queue/handoff was stopped. Invalid/aborted records and the old plan are retained and excluded.
- Explicit stationary v1 fixed that timeout path. Its sparse grids stayed stable but only reached 66–83% fill.
- Three denser grid pilots lost 9.99–15.24 nominal mL while stationary. Their records and plan remain retained and excluded. The settled-seed initializer replaces the unrun dense cases; it does not change evaluation fluid parameters.

## Resource isolation and reproducibility

Only the immutable `aiworker_pouring_4070` container is used: ID `0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`, GPU UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, four-core quota, CPU set 16–23, shares 256, 16 GiB memory/swap. The existing guard yields to foreign 4070 compute jobs by stopping only this container. The separate 4090 job is not stopped or reconfigured. Snapshots do not establish zero shared-host overhead.

Native HDF5 recordings, coefficients, checkpoints and private assets remain outside Git. Small provenance/reports and source are committed to the independent private repository. Because Mac disk space is limited, new raw recordings remain on Ubuntu. Media tools fetch one temporary source at a time, verify bytes/SHA, compose or inspect it, then remove only that temporary download. Scalar metadata is backed up locally. This is not a full local raw-data backup.

The review pipeline audits all camera decodes/timestamps and inventory partitions, independently recomputes all thirty scalar outcomes, checks source/weight/resource identities, and renders three complete labeled reels. Development review includes eight task/wrist/overview frames and ten return-sequence frames per attempt. The withdrawal diagnostic measures upright, near-stationary dwell over the cup and post-release recontact. Selected-frame review is not continuous manual playback.

## Preserved development lineage

| Round | Strict passes | Zero-loss attempts | Main change / retained limitation |
|---|---:|---:|---|
| r0 | 7/8 | 8/8 | Initial settled-level fit and direct return. One full-stop confirmation failure; one earlier recorder abort retained separately. |
| r1 | 7/8 | 7/8 | Full-stop scalar target reduced to 0.9294295. One unconfirmed full stop with overflow. |
| r2 | 4/8 | 4/8 | Earlier coarse brake at 0.82 and up to 13 pauses. Four loss cases totaling nine particles (0.052488 nominal mL); all eight return. |
| r3 / V24 | 8/8 | 8/8 | Training-only settled-pause supervision, 5,151-parameter refit, 8 mm late lip shift. Reviewed before the fresh thirty-test freeze. |

These are development/selection results, not additional fresh-test successes. All 32 complete development trajectories remain in the native audit, with 144 selected native frames reviewed per round. The r2 cases 03/06/07 each lose two particles, and 05 loses three; case 05 also retains an unconfirmed stop. Their scalar/visual diagnoses remain under `evidence/finish-A/development-r2-*`. The late lip shift does not directly change the 80° settling pose implicated in r2 case 06, and one different late-empty loss remains in the fresh benchmark.

The initial spatial fit took 24.86 seconds on the 4070, with 217/54 train/validation episodes and 6,247/1,476 images. Its 126-coefficient ridge model plus 139→64→16→1 residual contained 10,143 parameters (checkpoint `f26ff47868da35f47c9b8a6011132b25211a0cfd9c85522a0caa1468b65647ab`). Subsequent scalar candidates are archived under native `candidates/v23-r0/`, `v23-r1/` and `v23-r2/`; local binary candidates remain ignored. The selected V24 fit and its distinct training supervision are described above. An initial V24 refit attempt failed a metadata-preservation assertion before saving/deploying a candidate. Corrected alias handling preserves historical metadata; successful `refit-r3-attempt2` remains separate from that first attempt.

The original recorder interruption was infrastructure failure, not a task outcome: `FN30_A_dev_05_r0` ended at 44.755 simulated seconds during retreat due to approximately 1 microsecond of camera-clock skew. Its incomplete HDF5, 893 transitions, 5,358 decoded images and original error remain catalogued in `infrastructure-aborts.json`. The bounded repair accepts at most 100 microseconds of skew while retaining raw monotonic timestamps; a stale 1 ms view is rejected. Only that interrupted development case was repeated as `FN30_A_dev_05_retry1_r0`. None of the thirty reserved tests was retried.

The read-only timing helper reproduces independently diagnosed r2 loss times and retains their original outcomes. `failure-diagnostic-sources.json` and `failed-stop-gates-diagnostic.json` preserve the earlier two-case diagnosis during evaluation; the final five-case diagnosis is now bound to the complete audit. Reporting changes did not modify the frozen controller or its pass/fail rules. In particular, a missing return has null motion measurements, and an early abort with no tipping observations is not described as proven physical grip loss.

## Reproducing the completed evidence

Restore external scalar/model/native recording artifacts first; Git contains source and small reports only. These commands read completed evidence without queuing tests or fitting:

```sh
.venv/bin/python tools/summarize_finish_benchmark.py
.venv/bin/python tools/analyze_finish_outcomes.py \
  --report evidence/finish-A/final-benchmark.json \
  --integrity evidence/finish-A/recording-integrity.json \
  --output /tmp/aiworker-finish-timing-check.json
.venv/bin/python tools/diagnose_finish_stop_gates.py \
  --sources evidence/finish-A/failure-diagnostic-sources-final.json \
  --model evidence/finish-A/model-summary.json \
  --output /tmp/aiworker-finish-final-stop-gates-check.json
```

Use a new output filename if a diagnostic already exists. Native audit inputs, video mappings and reporting-helper hashes are bound in `review-summary.json`. Video manifests preserve every native-to-output frame mapping. The original full-scale data remain on Ubuntu; local scalar copies and the existing lossless historical archives are retained.
