# AIWorker — Pouring & Al-Jazari

Native Isaac Sim experiments for a stationary ROBOTIS FFW-BG2 learning to grasp a bottle, pour a colored drink, and stop near full or when empty. The historical companion explores Al-Jazari's drink-serving automata.

**Latest completed benchmark, November 3, 2026:4/6 fresh autonomous tests completed the full task; all six had zero measured spill.** All three full-cup cases passed at90.5–91.1%final rim height. Two short-bottle cases failed: an overly strict visual grasp check caused a safe abort, and an empty detector timed out. The older nominal-volume score is1/6. Reliability remains unmet; these are simulation results.

The controller uses visual servo grasping with encoder/lift confirmation, a supervised cup-height CNN, demonstration-fitted contextual motion, and programmed settled-level stop/return sequencing. Runtime inputs are RGB, measured joints, own history and fixed calibration. [Current study/results](docs/GRASP_AND_FILL_STUDY.md), [JSON](evidence/guard-A/final-benchmark.json). The follow-up detection repair is in progress; its results are not yet available. The prior [visual-stop study](docs/VISUAL_STOP_STUDY.md) remains archived.

Only the guarded RTX4070 was used, under fixed CPU/RAM limits. No foreign job was stopped or modified; the separate4090 process remained present throughout this frozen block. [Runtime evidence](evidence/guard-A/runtime-comparison.json).

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
