# AIWorker pouring

Native Isaac Sim experiments for a stationary ROBOTIS FFW-BG2 learning to grasp a bottle, pour a colored drink, and stop near full or when empty. The historical companion explores Al-Jazari's drink-serving automata.

**In progress — independent color/high-fill/moving views:** three baseline color candidates failed the unchanged accuracy gates; none is activated. All nine high-fill calibration scenes pass at 94.65–95.66% measured height with zero loss. All 18 additional low-fill and 18 moving validation scenes pass physical recording checks, but moving cup localization fails 7/18 accuracy checks. A supervised rim repair is in progress on the guarded 4070. Autonomous pours remain disabled for this successor study. [Current method and evidence](docs/DYNAMIC_COLOR_STUDY.md).

**Latest cup-size perception result:** supervised level estimation retrained on the RTX 4070 passes all declared development gates and **9/9 fresh reserved static tests** across nine known cup sizes. Mean/max held-out level errors are **0.66 / 1.42 percentage points**. The active pouring controller is unchanged; resized autonomous pouring remains gated on higher-fill, dynamic and independent-color checks. [Method and evidence](docs/LEVEL_SIZE_STUDY.md).

**Current practical tolerance:** the user accepts two lost particles and provisionally allows up to five per attempt. The completed batch is **30/30 within this tolerance**, with all other checks unchanged; its original zero-spill score stays **28/30**. This is a post-hoc scoring interpretation, not a new run or model improvement. [Supplemental score](evidence/side-return-A/test-r1-small-spill-tolerance.json).

**Completed November 9: 6/6 accepted development pours, then 28/30 fresh reserved tests accepted on the guarded RTX 4070.** All thirty pass stopping-margin, simultaneous side-return, placement, release and withdrawal checks. All fifteen full-cup tasks pass, settling at 92.87–96.19% rim height; all fifteen empty-bottle tasks finish with zero source particles. Two empty cases each lose two particles before stopping and remain failures: **the zero-spill target is not achieved**. The bottle now turns upright while moving to a feasible side placement, preferring robot-right, with no sampled upright-over-cup pose. Learned models are retained; this study is physics/control calibration, with no new gradient fit or RL. [Final study](docs/SIDE_RETURN_STUDY.md), [verified closure](evidence/side-return-A/test-r1-closure.json), [status](STATUS.md).

All thirty complete attempts are in three labeled local videos: [attempts 01–10, 3:54](evidence/side-return-A/test-r1-part1.mp4), [11–20, 3:47](evidence/side-return-A/test-r1-part2.mp4), [21–30, 3:53](evidence/side-return-A/test-r1-part3.mp4). Passes play at labeled 2× speed, failures at 1×, without internal cuts. [Normal-speed development returns, 1:03](evidence/side-return-A/development-r1-returns.mp4) show the six accepted return/set-down/release sequences. Full decoding and 270 native-camera comparisons pass for the reserved reels; 720 selected native frames and 96 selected encoded frames were actually inspected. Media stay outside Git; their manifests, hashes and review evidence are versioned. Earlier r0 placement failures remain preserved.

**Latest completed perception-stage evaluation: 4/6 reserved passes.** Both full cups reach at least 90% final rim height; two empty-bottle attempts each lose two particles. All six finish return/release. That archived path turns upright above the cup before moving away; the completed side-return study replaces it. Those results and failures remain preserved. [Outcomes](evidence/perception-A/test-r3-outcomes.json), [reviewed video manifest](evidence/perception-A/test-r3-video.json).

**Cup-size pilot and recovery development are complete:** nine resized cups pass revised low-fill containment checks, but the frozen detector confirms only 4/9 views. Six reused-layout regressions finish 3/6 strict passes, 5/6 zero measured loss and 6/6 returns/releases. Source ambiguity, full-stop prediction and one spill remain unresolved. Learned weights are unchanged; resized autonomy is not accepted. [Study and next action](docs/SIZE_CURRICULUM.md), [verified stage summary](evidence/size-A/stage-summary.json). The completed held-out benchmark below remains immutable.

**Earlier V24 benchmark, November 8, 2026: 25/30 fresh tasks passed independent scoring.** V24 removed the programmed stationary upright dwell; its path turns upright above the cup, returns the bottle to its starting position, places, releases and withdraws the hand. Twenty-nine attempts complete the return/release sequence; the completed post-pour upright/transfer takes 3.10–3.15 seconds, with at most 0.05 seconds stationary upright dwell over the cup. One empty-bottle attempt still fails to trigger return. Twenty-nine attempts have zero measured loss; one loses two particles (0.011664 nominal mL). The other failures are two unconfirmed full stops and an early tracking abort. Full-target tasks pass 12/15, empty-target tasks 13/15. This combines a supervised settled-level refit (238 training / 54 validation episodes, 5,151 parameters, 33.1 seconds on the RTX 4070) with an 8 mm late-pour lip adjustment and programmed terminal motion. All thirty ran once with frozen controls; all five failures remain retained. [Method and failures](docs/FINISH_STUDY.md), [verified results](evidence/finish-A/final-benchmark.json), [return measurements](evidence/finish-A/test-withdrawal.json). All three full native reels pass decoding and 270 camera comparisons; 96 selected encoded frames/cards are visually inspected. [Final review](evidence/finish-A/review-summary.json).

V24 local videos: [tests 01–10](evidence/finish-A/thirty-finish-tests-part-1.mp4), [tests 11–20](evidence/finish-A/thirty-finish-tests-part-2.mp4), [tests 21–30](evidence/finish-A/thirty-finish-tests-part-3.mp4). Every failure is included. Media are external to Git; manifests, hashes and review evidence are versioned.

**Previous wider/swapped benchmark, November 6, 2026: 18/30 fresh tasks passed independent scoring.** All thirty verified grasp, transfer and return/release. Full-cup tasks passed 5/15; empty-bottle tasks passed 13/15. In particular, swapped near-full tasks passed **0/7**, despite successful pickup. Twenty-six attempts had zero measured loss; four lost seven particles in total. Six full cups were underfilled and three full stops remained unconfirmed; one underfill also spilled. These categories overlap. The localization repair succeeded across this benchmark, but pouring and stopping are not yet reliable over the larger placement range. All 99 study recordings and 359 sidecars pass native integrity and local checksum verification. All three labeled videos are complete: full decoding and 270 camera comparisons pass, and 96 selected encoded frames were manually inspected. [Study and method](docs/WIDE_TABLE_STUDY.md), [verified results](evidence/wide-A/final-benchmark.json), [all thirty CSV rows](evidence/wide-A/final-benchmark.csv), [final review](evidence/wide-A/review-summary.json).

Local videos retain every attempt and failure: [tests 01–10, 4:07](evidence/wide-A/thirty-wide-tests-part-1.mp4), [tests 11–20, 3:32](evidence/wide-A/thirty-wide-tests-part-2.mp4), [tests 21–30, 3:27](evidence/wide-A/thirty-wide-tests-part-3.mp4). The media files are external to Git; their manifests, hashes and review reports are versioned.

The V22 controller evaluated in that benchmark uses RGB template localization with temporal consensus, a **340-coefficient settled-height ridge regressor**, a **48,001-parameter wrist CNN** for empty detection, and motion behavioral cloning from **43 training demonstrations**, constrained by IK and programmed grasp/stop/return logic. Final fitting took 288.03 seconds on the RTX 4070. This is a hybrid supervised controller, not reinforcement learning or end-to-end motor learning. All thirty tests ran once with unchanged source and weights; no test data entered fitting.

## Earlier benchmarks

**Expanded-position benchmark, November 4, 2026: 23/30 fresh autonomous tasks passed.** The RTX 4070 refit used positions three times wider on each of four axes: cup X/Y ±24 mm and bottle X/Y relative to the cup ±27/12 mm. Full targets passed 14/15 and empty targets 9/15. Two pours spilled a total of nine particles (0.052488 nominal mL). The other 27 pours had zero measured loss; one layout had no pickup or transfer. The model, placement plan and runtime were frozen before testing. [Study and failure details](docs/WORKSPACE_STUDY.md), [verified results](evidence/workspace-A/final-benchmark.json), [CSV](evidence/workspace-A/final-benchmark.csv).

**Completed narrow-workspace benchmark, November 4, 2026: 30/30 fresh tasks passed.** Temporal localization, additional bottle templates, slower initial tilt and bounded top-ups address the previous two localization failures and one spill. After fitting on the RTX 4070, all thirty new tasks verified grasp, liquid transfer, the correct stop, and complete upright return/release. All fifteen full-cup tasks ended at **91.33–92.87% of rim height**; all fifteen empty-bottle tasks ended with **zero source particles**.

All thirty had zero final outside particles and zero sampled near-table loss. Thirteen had transient nonairborne outside readings, peaking at three particles (0.017496 nominal mL), which returned to zero loss. Measurements run at 20 Hz while physics runs at 400 Hz; this is not proof of splash-free motion or no between-sample excursions. The separate legacy nominal-inventory score is **16/30**: it measures nominal volume, whereas the declared full-cup goal measures surface height. All thirty ran once, without tuning or retries. [Study and method](docs/RELIABILITY_STUDY.md), [verified JSON](evidence/reliable-A/final-benchmark.json), [all 30 CSV rows](evidence/reliable-A/final-benchmark.csv), [outcome analysis](evidence/reliable-A/failure-analysis.json). The previous [27/30](docs/SETTLED_LEVEL_STUDY.md), [25/30](docs/ROBUSTNESS_A_30.md) and [6/6](docs/GRASP_VOTE_STUDY.md) evaluations remain separate and unchanged.

The earlier narrow-workspace controller combined visual servo grasping, frozen supervised vision, a 16-coefficient settled-level ridge regressor fitted on **98 training / 28 validation episodes**, movement imitation fitted on **35 / 8 demonstrations**, and feedback control. Its preceding height model used 58 / 18 episodes. Runtime inputs are RGB, joints, own history and fixed calibration. Those results concern material A within that smaller calibrated workspace; physical transfer is untested. They are separate from the larger, swapped-position benchmark above.

The guarded 4070 container retained its four-core CPU quota and 16 GiB RAM cap throughout fresh evaluation. No foreign job was signaled or reconfigured; snapshots do not establish uninterrupted execution of other workloads. The model, 22 runtime sources, original asset and all **44 development/test recordings plus 176 sidecars** were verified; no partial recordings remain. All five test camera sheets were reviewed. That narrow benchmark and the later expanded-position benchmark are both complete. [Resource evidence](evidence/reliable-A/runtime-comparison.json), [recording audit](evidence/reliable-A/recording-integrity.json), [final review](evidence/reliable-A/review-summary.json).

This private repository versions source, configuration, documentation and small result/provenance files. Recordings, videos, weights, USD assets, vendor dependencies and credentials remain external. **A clone requires those external runtime dependencies to reproduce native runs.** See [artifact locations and Git workflow](docs/VERSION_CONTROL.md), [runtime setup](runtime/README.md) and [current status](STATUS.md).

Quest video and right-controller input were restored and verified earlier. The [Quest runbook](docs/QUEST_RESTORE.md) retains those setup instructions. Historical media/evidence links resolve only when the untracked artifacts have been restored.

## Start here

- [Recovered setup, evidence and resume point](docs/RECOVERED_SETUP.md).
- [Recovery checklist and dry-rehearsal plan](docs/RECOVERY.md).
- [Current ROBOTIS workflow and source links](docs/ROBOTIS_WORKFLOW.md).
- [First demo and measurement protocol](docs/FIRST_DEMO.md).
- [Recording schema](docs/RECORDING_SCHEMA.md).
- [Historical fact check](docs/HISTORY.md).
- [45–60 second script and shot list](docs/VIDEO.md).
- [Status and next action](STATUS.md).

## Offline trial logger

Python 3.9+; standard library only; macOS/Linux. No installation, GPU, network, ROS, or robot commands. Each JSONL entry is one attempted cycle. The logger evaluates fixed `pour-v1` thresholds and separates evidence kinds, control modes, phases, setups, software, and policies.

From this directory:

```sh
python3 -m unittest discover -s tests -v
python3 tools/trial_log.py validate examples/synthetic-wet.json
python3 tools/trial_log.py record examples/synthetic-wet.json --log data/synthetic-smoke.jsonl
python3 tools/trial_log.py summary data/synthetic-smoke.jsonl
```

The example is invented data. Its passing arithmetic is **not a successful robot pour**. Repeating the `record` command rejects its duplicate trial ID; use `validate` to repeat the calculation without appending.

Create an incomplete form for a future simulator dry trial:

```sh
python3 tools/trial_log.py template --trial-id SIM-001 --series-id SIM-DRY-A --evidence-kind simulation --phase dry --mode scripted > data/sim-001-form.json
```

Use a fresh filename when redirecting. Fill the blank metadata and explicit boolean flags after the attempt; leave all mass fields `null` for dry trials. Set the timestamp to the actual attempt start. Supply a real artifact reference, exact robot asset identifier and software version. Then:

```sh
python3 tools/trial_log.py validate data/sim-001-form.json
python3 tools/trial_log.py record data/sim-001-form.json --log data/simulation-trials.jsonl
```

`validate` returns zero for a well-formed failed attempt: read `passed` and `reasons`. Malformed input returns exit code 2. Aborted or unmeasured attempts remain in the denominator. Missing measurements never become zero. `record` rejects duplicate IDs, changed conditions within a series, out-of-order series timestamps, or a corrupt existing log. `summary` uses the latest five attempts per condition group for the provisional physical repeatability rule. It never grants that rule to synthetic, simulated, or dry records.

The tool checks the operator's entries, not artifact existence, sensor calibration, or truthful labeling. Keep raw evidence, log every attempt, and make a new series after changing the setup/controller. It cannot establish reliability from five trials or verify safety.
