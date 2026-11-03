# ROBOTIS workflow — checked October 28, 2026

These are current official descriptions, not installed-software findings. The user's July environment may differ. No ROBOTIS stack was installed or launched here.

## Components and decision

| Component | Role for this project | Official source |
| --- | --- | --- |
| `ai_worker` | ROS 2 descriptions, bringup, hardware integration and teleoperation | [Repository index](https://docs.robotis.com/docs/systems/aiworker/resources/open_source/), [source](https://github.com/ROBOTIS-GIT/ai_worker) |
| Cyclo Intelligence | Demonstration recording, dataset preparation and policy inference UI | [Overview](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/overview/), [source](https://github.com/ROBOTIS-GIT/cyclo_intelligence) |
| Cyclo Control | Motion generation/filtering; not a trained pouring policy | [Guide](https://docs.robotis.com/docs/systems/aiworker/advanced_features/cyclo_control/), [source](https://github.com/ROBOTIS-GIT/cyclo_control) |
| Cyclo Lab | Isaac Sim/Lab robot assets and example tasks | [Guide](https://docs.robotis.com/docs/systems/aiworker/simulation/isaac_sim_lab/), [source](https://github.com/ROBOTIS-GIT/cyclo_lab) |

ROBOTIS designates Cyclo Intelligence as actively supported and Physical AI Tools as legacy with no further updates. For new hardware data collection, prefer the supported path. Preserve a working older simulation environment while checking conversion compatibility. [Learning overview](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/overview/)

## Demonstration → policy

1. **Inventory and bringup.** Identify follower and leader, installed revisions, available cameras and controller ownership. The current teleoperation guide distinguishes BG2/SG2 followers from the LG2 leader. It specifies Zenoh for AI Worker 2.0.0 onward; Cyclo Manager may already run the daemon. Do not duplicate it. Model-specific startup and pause behavior must be checked against the installed version before enabling motion. [Teleoperation](https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/teleoperation/)
2. **Connect the recorder.** The documented Cyclo Intelligence setup uses Docker 24+, Compose, 35 GB available storage and port 7080; the web UI/recording/data tools do not require a GPU, while trained-model inference does. Its ROS 2 network connection is configured for the existing Zenoh daemon. Confirm the actual machine before installing or changing anything. [Setup](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/setup/)
3. **Record one task.** Start robot and camera publishers first. Select the matching robot configuration, inspect Camera Views and Topic Monitor, then record rosbag2 episodes. Only topics in `shared/shared/robot_configs/<robot>_config.yaml` are recorded. Verify both observations and command/action topics. Use one instruction such as “Pour water from the held bottle into the fixed cup, then return the bottle upright,” with zero subtasks for the continuous first demo. [Data Recording](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/data_recording/)
4. **Inspect and convert.** Replay camera/joint data, preserve originals, and exclude unusable attempts from a separate training copy while retaining all outcome logs. Data Tools can merge tasks and convert to LeRobot 2.1 or 3.0. Use a target FPS supported by the slowest actual camera stream. Output paths in the documented container are `/workspace/rosbag2` and `/workspace/lerobot`. Conversion alignment is not proof of adequate synchronization; inspect it. [Data Tools](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/data_tools/)
5. **Train only after data review and compute allocation.** Training happens outside Cyclo Intelligence. ROBOTIS recommends ACT as an initial choice when uncertain; ACT learns from the collected dataset rather than automatically receiving a general pouring skill. Pin the dataset, model, environment and inference backend together. No training job is part of this preparation; shared Ubuntu GPU jobs remain untouched. [Training guide](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/model_training/)
6. **Evaluate before deployment.** The documented `3D Sim Deploy` checks one action chunk from the current robot state and blocks physical commands. It is not a fluid simulation or proof of a whole pour. Stop teleoperation before enabling real inference. Check backend compatibility: the current guide restricts its prebuilt ACT/GR00T images to Blackwell-based amd64/arm64 environments, so an existing Orin or other GPU cannot be assumed compatible. Match the installed backend and checkpoint before any dry deployment. [Inference](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/model_inference/)

## Cyclo Control, where useful

The guide exposes Cartesian `MoveL`, joint-space `MoveJ`, and bimanual variants using a QP controller. It considers joint limits and self-collision, but explicitly warns that self-collision avoidance is not guaranteed. It does not establish bottle/table/cup collision avoidance or spill prevention. For this task, keep one arm and one fixed cup initially. A later script may use locally verified poses and the appropriate controller; no example coordinates are copied into an executable pour. Confirm actual frame names, joint order, gripper semantics, topic remapping and the single active command publisher first. [Cyclo Control](https://docs.robotis.com/docs/systems/aiworker/advanced_features/cyclo_control/)

## July simulation compatibility

The current Cyclo Lab guide lists Isaac Sim 5.1.0 and Isaac Lab 2.3.0 in its Docker image. It includes a BG2 reach example and SG2 pick-and-place recording/conversion examples, but still routes those examples through `physical_ai_tools`. These examples establish available starting points, not a ready-made pouring task or compatibility with July's project. Preserve the recovered versions and inspect the actual recorder/action schema before migration. [Isaac guide](https://docs.robotis.com/docs/systems/aiworker/simulation/isaac_sim_lab/)

BG2 has no mobile drive in the current hardware comparison; SG2 has swerve drive. This makes BG2 a plausible interpretation of “stationary base.” An SG2 asset fixed to the world would also appear stationary. Confirm the USD/config, not the visual impression. [Hardware comparison](https://docs.robotis.com/docs/systems/aiworker/specifications/hardware/)
