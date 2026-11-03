# AIWorker — Pouring & Al-Jazari

Native Isaac Sim experiments for a stationary ROBOTIS FFW-BG2 learning to grasp a bottle, pour a colored drink, and stop near full or when empty. The historical companion explores Al-Jazari's drink-serving automata.

**Latest measured result, November 3, 2026: 4/6 fresh autonomous tests completed the entire task with zero measured spill.** Five grasps and five return/set-down/release sequences completed. One grasp failed and spilled; one full-stop prediction was early and left an 83.2% cup level. The reliability target remains unmet. These are simulation results; physical-robot transfer is untested.

The controller uses hierarchical supervised imitation: visual behavior cloning for grasp/lift, a liquid-state CNN, contextual movement primitives fitted from demonstrations, and programmed stop/return/release sequencing. Runtime inputs are RGB, measured joints and the robot's own history; simulator inventory and object poses are labels/scoring information only. [Technical study and six-case results](docs/VISUAL_STOP_STUDY.md), [JSON benchmark](evidence/stop-A/final-benchmark.json).

Experiments used only the guarded RTX 4070, with fixed CPU/RAM limits. No foreign job was stopped or modified by this work. A 4090 process present before testing was absent at the final check; its exit reason is unverified. [Runtime evidence](evidence/stop-A/runtime-comparison.json).

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
