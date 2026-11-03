# Minimal demonstration and trial schema

Use two layers: a small outcome record for each attempted cycle and a time-series demonstration for learning. `tools/trial_log.py` implements only the outcome record. It does not record ROS, encode video, control a robot, convert LeRobot data, or validate a trained policy.

## Outcome JSON / JSONL

`examples/synthetic-wet.json` is the full machine-readable example. `template` produces all required keys with unknown observations left null. Missing/extra keys, invalid enum values, duplicate JSON keys, nonfinite/negative numeric values and ambiguous boolean strings are rejected.

| Fields | Meaning |
| --- | --- |
| `protocol`, `trial_id`, `series_id` | `pour-v1`; unique attempt ID; predeclared experiment block |
| `started_at` | ISO 8601 including offset; display/plan in America/Los_Angeles |
| `evidence_kind`, `phase`, `mode`, `mode_detail` | Physical/simulation/synthetic; dry/wet; control category and exact assistance/stages |
| `robot_model`, `setup_id`, `software_ref` | Identified robot/asset, immutable setup manifest, controller/repository revision or file hash |
| `policy_ref`, `episode_ref`, `artifact_ref` | Checkpoint hash/path for learned control; optional bag/HDF5 episode; video/log evidence reference |
| `duration_s`, `initial_water_g` | Measured duration including settling; initial water mass |
| `cup_before_g`, `cup_after_g` | Paired gross cup readings |
| `bottle_before_g`, `bottle_after_g` | Paired gross bottle readings |
| `spill_before_g`, `spill_after_g` | Separate spill collector readings |
| `cycle_complete`, `intervention`, `collision`, `object_drop`, `estop`, `spill_outside_tray` | Explicit observations for each attempt; never default an unknown to false |
| `notes` | Failure reason, reset details, deviations, fluid model or synthetic provenance |

Dry records use null for every mass field. Wet aborted records may have null measurements and will fail evaluation while remaining in the log. Missing duration also prevents a pass. Units are seconds and grams; derived values are mass, not a calibrated volume estimate. The timestamp on the synthetic fixture is illustrative.

## Session/setup manifest

For each immutable `setup_id`, retain a JSON/YAML manifest with: host/runtime versions; repository revisions and container digests; robot variant plus asset hash/serial; joint order; controller and action semantics; fixed/free base; bottle/cup IDs, geometry and initial poses; frame definitions and units; fixture photo; scale identity/resolution; camera config and intrinsics/extrinsics when used; control and camera measured rates; scene physics settings if simulated; and calibration provenance. Unknown values remain null and block any claim depending on them. A changed scene/controller/initial condition creates a new setup/series.

## Minimum time-series episode

| Signal | Minimum record |
| --- | --- |
| Identity | Trial ID ↔ original bag/HDF5 episode ↔ derived dataset index; one task instruction and evidence kind |
| Time | Monotonic sample index plus sensor/source timestamp and recorder timestamp; clock domain/offset method |
| Images | Task view showing bottle mouth/cup; timestamp, stream key, frame reference, dimensions, encoding, measured FPS; wrist view if available |
| Observations | Actual joint positions and gripper state; joint names/order and units; velocities/effort if available, explicitly absent otherwise |
| Actions | Commands issued, their timestamp and controller source; absolute versus delta, position versus velocity, frames/units, scaling and gripper convention |
| Events | Start, tilt, first flow if observable, upright, stop, reset and any intervention/incident |
| Outcome | The linked trial JSON with all attempts, including failed or unmeasured episodes |

Do not substitute measured joint positions for action commands without proving that this matches the intended training interface. Preserve the source action stream and any controller-filtered stream with distinct names.

Before collecting a dataset, record/replay one dry episode: confirm the configured image/state/action topics are present, timestamps are monotonic, stream gaps are visible, and alignment is within the measured application tolerance. A provisional goal is no more than one recorded frame interval of image/state skew; measure it rather than assuming equal FPS implies synchronization. Stop data collection to repair missing streams.

ROBOTIS's recorder only captures configured topics, and its Data Tools converts reviewed episodes to LeRobot datasets with aligned actions/observations. Check the actual configuration before recording and inspect converted examples before training. [Recorder](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/data_recording/), [conversion](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/data_tools/)

For an initial data-quality pilot, collect ten successful demonstrations plus retain every failed attempt's evidence. Ten is a proposed inspection batch, not a claim that ten examples will train a reliable policy. Split by complete episodes/sessions, never adjacent frames. Reserve a later untouched evaluation block; do not tune on it. Keep originals and use a separate training subset. Do not upload data by default. A simulator HDF5 recorder may require a different adapter from the physical rosbag2 route; recover July's schema before choosing one.


## Native Isaac Sim camera pilot — November 2, 2026

Material **A** is selected for bottle and cup. Both normal AIWorker pour buttons now record a scripted reference episode with two virtual pinhole cameras at 384 × 384 RGB and 20 Hz in simulated time. The fixed task camera observes the work area. The right-wrist view follows the measured wrist transform. Their lens attributes, calibration matrices and per-frame camera poses are saved. The policy interface is 11 absolute arm/gripper joint targets in radians held for 0.05 s, matching the recorded 20 Hz controller.

Each completed trial writes episode.hdf5 atomically from episode.partial.hdf5. Interrupted files remain partial and are excluded. The standard transition is obs(t), actions(t), next_obs(t+0.05). data/demo_0 contains obs, next_obs, actions, time, next_time, action_time, rewards, dones and a separate ground_truth group. Policy RGB/state keys are task_image, wrist_image, joint_pos and joint_vel. Time/reference/camera-pose fields are metadata. Exact liquid inventories, bottle pose, grasp state and stop labels belong only to supervision/scoring. Initial camera warm-up precedes the first saved observation.

The curated local pilot index is data/camera-A-pilot/manifest.json. It contains two visually reviewed, physically successful expert demonstrations and SHA-256 hashes. The action-replay episode is excluded from training. No episode split or trained policy is claimed from this small fixed-scene pilot. Current Robomimic is absent from this runtime, and the native task is not yet a registered Robomimic evaluation environment. A training adapter remains part of the next policy-training stage.

On Ubuntu, use either live AIWorker button or run the following command to start a recording.

```sh
curl -sS -H 'Content-Type: application/json' -d '{"command":"robot_full"}' http://192.168.68.60:8212/control
```

Use robot_empty for the small source and robot_replay to execute the latest successful saved joint actions. The task and wrist RGB previews are available at /camera/task.jpg and /camera/wrist.jpg on the same server.

Validate a completed recording inside the owned container. Replace EPISODE_PATH with the full /workspace/outputs/fountain/.../episode.hdf5 path reported by /health.

```sh
ssh kfuUbuntu docker exec -w /workspace/src aiworker_pouring_4070 /isaac-sim/python.sh validate_pour_recording.py EPISODE_PATH --export-views --output VALIDATION_PATH
```

Inspect exported initial, grasp, pour, stop and final camera images before using --approve-views. The validator checks every transition's image/state timestamps, image continuity, action dimensions/units and camera calibration fields. It rejects incomplete episodes, stale images and shifted actions. Native SDK interpolation warnings were observed around rendering. Accepted episode image timestamps passed the explicit physics-time comparison, with maximum skew below 0.25 microseconds. The diagnostic framing pilots remain preserved and excluded.

