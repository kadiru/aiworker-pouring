# AIWorker pouring

Native Isaac Sim experiments for a stationary ROBOTIS FFW-BG2 learning to grasp a bottle, pour a colored drink, and stop near full or when empty. The historical companion explores Al-Jazari's drink-serving automata.

**Active follow-up:** near-full prediction and a direct upright return, placement and hand withdrawal. Round 0 completes 7/8 strict passes with zero measured loss in all eight. Round 1 completes 7/8, retaining one swapped-layout overflow. Both rounds verify at most 0.05 seconds of upright stationary dwell over the cup versus the historical 1.1-second median, and no post-release gripper recontact. An earlier initial brake (82% versus 87%) and up to thirteen small top-ups are now running through eight r2 development checks. R2 is complete with 4/8 strict passes: three separate two-particle losses and one three-particle overfill with unconfirmed stop. All eight grasp, transfer and return; seven confirm the required stop. A level refit on the collected development views is prepared, along with an 8 mm late-pour lip adjustment that passes 38 geometry checks and 24 CPU checks. Neither new refit nor alignment is active yet; physics and natural return timing remain unchanged. The thirty reserved tests remain untouched. These development results do not replace the completed benchmark below. [Follow-up method and evidence](docs/FINISH_STUDY.md).

**Wider/swapped benchmark, November 6, 2026: 18/30 fresh tasks passed independent scoring.** All thirty verified grasp, transfer and return/release. Full-cup tasks passed 5/15; empty-bottle tasks passed 13/15. In particular, swapped near-full tasks passed **0/7**, despite successful pickup. Twenty-six attempts had zero measured loss; four lost seven particles in total. Six full cups were underfilled and three full stops remained unconfirmed; one underfill also spilled. These categories overlap. The localization repair succeeded across this benchmark, but pouring and stopping are not yet reliable over the larger placement range. All 99 study recordings and 359 sidecars pass native integrity and local checksum verification. All three labeled videos are complete: full decoding and 270 camera comparisons pass, and 96 selected encoded frames were manually inspected. [Study and method](docs/WIDE_TABLE_STUDY.md), [verified results](evidence/wide-A/final-benchmark.json), [all thirty CSV rows](evidence/wide-A/final-benchmark.csv), [final review](evidence/wide-A/review-summary.json).

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
