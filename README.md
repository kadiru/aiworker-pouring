# AIWorker pouring

Native Isaac Sim experiments for a stationary ROBOTIS FFW-BG2 learning to grasp a bottle, pour a colored drink, and stop near full or when empty. The historical companion explores Al-Jazari's drink-serving automata.

**Latest native benchmark, November 3, 2026: 27/30 fresh tasks passed (90.0%).** After localization-coverage correction and RTX 4070 fitting of a settled-level predictor, 28/30 tasks verified a grasp, transferred liquid and completed return/release. All 14 executed full pours reached 91.99–95.46% rim height; all 14 executed empty-source pours ended with zero source particles. Two tasks rejected localization before grasping. One full pour lost one particle (0.005832 nominal mL) and remains failed. Reliable localization and zero spill are not yet solved.

Strict scores: 13/15 full, 14/15 empty; legacy nominal-inventory score: 23/30. Twelve cases had nonairborne outside-particle readings: eleven returned to zero loss, while the final case retained one escaped particle. No-pour aborts are never counted as task successes. All 30 ran once without tuning. [Study and method](docs/SETTLED_LEVEL_STUDY.md), [verified JSON](evidence/settled-A/final-benchmark.json), [all 30 CSV rows](evidence/settled-A/final-benchmark.csv), [failure analysis](evidence/settled-A/failure-analysis.json). The [previous 25/30 evaluation](docs/ROBUSTNESS_A_30.md) and [prior 6/6 study](docs/GRASP_VOTE_STUDY.md) remain separate.

The controller combines visual servo grasping, frozen supervised vision, a 16-coefficient settled-level ridge regressor trained on 37 episodes with 12 validation episodes, demonstration-fitted motion and a feedback state machine. Runtime inputs are RGB, joints, own history and fixed calibration. It is not reinforcement learning or a newly trained end-to-end motor policy. Evaluation is limited to material A and the existing small calibrated workspace; physical transfer is untested.

The same guarded 4070 container and CPU/RAM caps were retained. No foreign job was signaled or reconfigured; foreign 4090 process IDs changed for unverified reasons. Model, 18 runtime sources, original asset and all 48 completed development/test recordings plus retained partials were checksum-verified. The lab is idle with its guard active. [Resource evidence](evidence/settled-A/runtime-comparison.json), [recording audit](evidence/settled-A/recording-integrity.json).

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
