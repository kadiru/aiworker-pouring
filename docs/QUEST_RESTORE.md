# Restored Quest teleoperation and pouring scene

Updated October 29, 2026, America/Los_Angeles. The current scene is the [five-cup liquid comparison](LIQUID_COMPARISON.md). The server is running. The encoder issue is repaired, and after a later signaling recovery two desktop CloudXR sessions connected and decoded video. The certificate-verified Quest WSS endpoint now delivers a video offer. The operator has now confirmed Quest video, and actual right-controller samples were observed with the left controller absent. A successful teleoperated pour remains to be tested. The earlier signaling stall cause is unresolved; do not call the recovery a permanent fix. See [encoder repair details](../runtime/encoder/README.md).

## Connect the Quest

1. Put the Quest 3 on the same LAN as Ubuntu, `192.168.68.60`.
2. In the **Quest browser**, open [the local CloudXR certificate page](https://192.168.68.60:48322/). The operator must handle its self-signed-certificate prompt on the headset. This page should display “Certificate Accepted.”
3. Open [NVIDIA's Isaac Teleop client](https://nvidia.github.io/IsaacCapture/client/v1.1.8/) in that browser. Enter **`192.168.68.60`**, without a port, and connect/enter the immersive session.
4. Keep [the AIWorker control page](http://192.168.68.60:8212/) visible on a computer. XR normally enables automatically at startup. The button then says **XR enabled** and is disabled; no extra click is required. If it says **Start XR**, click once to enable the XR profile. The left/right controller indicators should change from **waiting** when tracked packets arrive. XR enabled does not itself confirm that the headset is receiving video.
5. Hold the **right trigger** to move the right arm relative to the controller. Release and reacquire it to reposition your hand without moving the arm. The **right grip** closes the gripper; releasing it opens the gripper while control is active. Tracking loss resets the movement reference. Begin with small motions away from the cups.
6. **A** starts recording at the current pose. **B** saves the attempt and resets the scene. **X** discards the attempt and resets. The web page provides the same buttons, plus pause, scene reset and view reset. Saving is not a declaration that a pour succeeded.

The live page also shows right-wrist, head and overhead RGB views. **Test arm IK** runs a synthetic controller sequence and saves a clearly labeled test episode; it does not test the headset or attempt a cup grasp/pour.

Quest 1.1.8 attempts after the encoder repair still failed: signaling carried peer announcements and ACKs but no video offer. A diagnostic restart with `NV_CXR_SERVER_LOG=debug` and proper acknowledged checks restored offers and desktop video decoding. The installed native runtime binaries are byte-identical to the 1.1.8 wheel; no upgrade was needed. The desktop test client is disconnected. The subsequent Quest session lasted about six minutes and produced right-controller input; [headset evidence](../evidence/quest-headset-connected.json). `xr_display_enabled` can remain true after disconnect and does not establish a live headset connection. See [streaming evidence](../evidence/quest-stream-negotiation.json).

## What is running

- Remote directory: `/home/kadir/aiworker-pouring-4070`, mounted as `/workspace`.
- Owned container: `aiworker_pouring_4070`; current verified ID `77c51242a32de7a7494f3f19c4b3b066c401fe9fd9c578c3ca5b164ef4608182`.
- **Isaac Sim 5.1.0-rc.19**, the existing `/home/kadir/isaac-sim` installation mounted read-only. The container base image is `nvcr.io/nvidia/isaac-sim:6.0.1`; the executable used by this profile is the mounted **5.1** runtime.
- Recovered **Isaac Lab 2.3.0** and ROBOTIS `robotis_lab` source. **Isaac Teleop 1.0.193 / CloudXR Runtime 6.1.0** live in a separate project venv.
- One stationary **FFW-BG2**, 25 movable joints, recovered right-arm relative 7D DLS IK and binary four-joint gripper. Original robot collision decompositions are restored. One redundant world fixed joint is disabled in the new scene layer. Robot-body gravity remains disabled as in the recovered setup.
- Five graspable cups, each **60.32 mm body diameter × 120 mm height**, with equal nominal 62 mL fills. All use the original source-cup geometry at scene scale `(0.58, 0.58, 0.75)`. The original USD assets remain unchanged. The earlier CPU open-gripper check found 43.87 mm total body clearance; no successful grasp is claimed.
- **7,539 PBD particles across five independent fluids**. Blue: 2.8 mm spacing / viscosity 0.01; white: 3.6 mm / 0.01; orange: 4.4 mm / 0.01; cyan: 3.6 mm / 0.001; purple: 3.6 mm / 100. Viscosity is a dimensionless simulator coefficient. Separate systems do not physically mix. [Full settings and comparison protocol](LIQUID_COMPARISON.md).
- Continuous clear-water isosurfaces replace individual spheres. Translucency and reflections are enabled. Raw physical positions remain the recording/measurement source; rendered mesh volume is not a liquid-mass measurement. Particle conservation, edge escape, possible tabletop crossings and mesh coverage are reported separately.
- RGB cameras: right wrist, head, overhead, **244×244**. No left-wrist or depth stream is recorded by this profile.
- No learned policy, successful cup grasp, source-to-target liquid transfer, or physical robot trial is claimed.

The historical `khemoo.examples.fluid_in_cup` extension was absent. `src/pour_fluid.py` implements the replacement PBD fluids. CPU tensors and disabled Fabric permit physical particle readback; particle dynamics still run on the GPU. The local physics wrapper retains native simulate/fetch for intermediate substeps and runs the full PhysX update on each rendered action, so fluid surfaces refresh without rebuilding them ten times per displayed frame. Installed Sim/Lab sources remain untouched.

Physics remains **200 Hz**, ten steps per configured 20 Hz action, with 16 particle iterations. These are simulated-time rates, not achieved wall-clock throughput. Surface capacity is bounded per system to 131,072 vertices and 262,144 triangles; sparse-grid budgets are 4,096 for the fine liquid and 2,048 for each other liquid. Camera updates are every 0.1 simulated seconds; repeated action-aligned RGB frames are possible. Preview requests are limited to five fps. See the latest comparison evidence for measured throughput; do not raise shared-host caps to improve it.

**Spill 1–5** resets the scene and runs a standardized settled-fill release, holding robot commands. **Reset scene** restores all five fills and normal teleoperation. These are synthetic drop tests, not robot pours. Saves during a synthetic test are marked `synthetic_validation`. Normal A/B recordings remain `quest_or_browser`, with outcomes unverified.

## Humanoid priority

Only RTX 4070 UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6` is visible. RTX 4090 is excluded. Limits are four CPU cores of quota, affinity 16–23, CPU shares 256, nice 10, 16 GiB RAM/no extra swap, 1 GiB shared memory and 1,024 PIDs. The updated scene sampled 3.67–3.82 GiB container RAM and about 2.3–2.4 CPU cores before the video test; process GPU memory changes when a stream connects. Shared-host overhead is not zero.

`launch_container.py` refuses an occupied 4070. Its host guard polls every two seconds and stops **only the immutable owned AIWorker container ID** if a foreign compute process appears there or inspection fails three times. It never stops the other job or automatically restarts AIWorker. The guard has five mocked boundary tests; no competing GPU job was launched to test it. Guarding cannot prevent the brief overlap before detection/shutdown.

The user authorized coordination with “Low-seat sitting repair — clear continuation,” which allowed AIWorker use of the 4070. This restoration did not alter humanoid jobs, project files, environments, drivers or host installations. The old GPU-0 `isaacsim6_home` container remains stopped with its September 5 start timestamp unchanged.

## Recording and evidence

Data are stored remotely under `/home/kadir/aiworker-pouring-4070/datasets/quest_pouring_YYYYmmdd_HHMMSS.hdf5`, with an adjacent `.events.jsonl` file. Each A/B recording contains actions, processed actions, initial and stepped robot/cup states, end-effector observations, wall timestamps, three RGB image streams and, in fluid mode, all particle positions. The five arrays are concatenated as `[frames, 7539, 3]`; episode attribute `fluid_conditions_json` gives the condition parameters and half-open slices. Start at a useful pose before pressing A. At 1,200 samples the recorder saves and resets to bound memory use.

Every saved attempt has `evidence_kind=simulation`, `outcome_verified=false`, `success=false`, and `success_label=not_evaluated`. `success=false` is a conservative storage flag, not an evaluated failure; use `success_label` when interpreting outcomes. Synthetic probes have `input_source=synthetic_validation`; normal operator runs currently use `quest_or_browser`. Do not mix synthetic validation recordings into a teleoperation dataset without explicit filtering.

The active HDF5 file is open/locked. After export, a flushed point-in-time readable copy is saved as `/home/kadir/aiworker-pouring-4070/outputs/quest-last-export.hdf5`. It is overwritten on the next export; copy it elsewhere within the project when retaining a particular demonstration. The timestamped dataset is the durable session record. Do not disable HDF5 locking to inspect the active writer.

Earlier single-cup evidence (superseded by the [five-cup comparison](LIQUID_COMPARISON.md)):

- [Cup/gripper USD measurements](../evidence/cup-geometry-inspection.json).
- [Three wet resets and surface mesh](../evidence/cup-water-validation.json): all 1,331 particles contained; visible water mesh around 1,170 vertices/2,340 triangles, sphere prototypes hidden.
- [Synthetic arm/gripper check](../evidence/cup-water-control-check.json): arm excursion 0.1003 rad, gripper excursion 1.0000 rad, return error 0.0000578 rad; no grasp or pour attempted.
- [Updated recording](../evidence/cup-water-recording-validation.json): finite particle arrays `[160,1331,3]` and all three RGB streams; labeled synthetic, outcome unverified.
- [Pause/isolation check](../evidence/cup-water-runtime-verification.json), [final startup/stream state](../evidence/cup-water-final-check.json). The full physics/recording checks precede the final streaming-only launcher refinement; container IDs are retained in each report.

Earlier baseline evidence (534-particle scene):

- [Actual IK and gripper response](../evidence/quest-dry-validation.json): 160 synthetic controller frames, arm peak 0.09998 rad, gripper peak 1.00000 rad, arm return error 0.0000674 rad; three reproducible dry resets.
- [Liquid and camera checks](../evidence/quest-fluid-camera-validation.json): three wet resets, all 534 particles contained, nonblack live camera views.
- [HDF5 recording checks](../evidence/quest-recording-validation.json): finite actions `[160,7]`, joints `[160,25]`, particle positions `[160,534,3]`, three RGB arrays `[160,244,244,3]`; temporal image changes and 3.78 cm particle movement.
- [Final runtime check](../evidence/quest-runtime-verification.json): pause freezes physics/joints/fluid while images update, resume advances physics, GPU isolation and resource caps match.
- [Controller unit checks](../evidence/quest-retargeter-tests.txt): five CPU checks cover trigger/deadman reacquisition, tracking loss, malformed input, clamp/gripper and equivalent quaternion signs. These do not substitute for a Quest test.
- [Live control-page screenshot](../evidence/quest-pouring-live.png).

## Operation and restart

Read status without changing anything:

```sh
ssh kfuUbuntu 'curl -fsS http://192.168.68.60:8212/health'
ssh kfuUbuntu 'docker logs --tail 60 aiworker_pouring_4070'
ssh kfuUbuntu 'tail -n 20 /home/kadir/aiworker-pouring-4070/outputs/cloudxr-launch.log'
```

The launcher only creates a container when that name is absent. To replace an existing owned container, read and verify its current ID, then pass that exact ID to the identity-checking replacement tool. The following ID is the final verification ID; it intentionally fails after a different replacement:

```sh
ssh kfuUbuntu 'docker inspect --format "{{.Id}} {{json .Config.Labels}}" aiworker_pouring_4070'
ssh kfuUbuntu 'python3 /home/kadir/aiworker-pouring-4070/restart_owned.py --expected-id 77c51242a32de7a7494f3f19c4b3b066c401fe9fd9c578c3ca5b164ef4608182 --profile quest --fluid'
```

If no container with that name exists:

```sh
ssh kfuUbuntu 'python3 /home/kadir/aiworker-pouring-4070/launch_container.py --profile quest --fluid'
```

Omit `--fluid` for an empty-cup Quest rehearsal. `--profile baseline` is the older simplified Sim 6 preview, not the full teleoperation setup. Do not use `docker start`, which bypasses launch preflight. If the guard yielded to another job, wait until the humanoid owner releases the GPU before relaunching.

First use of the original collision geometry took about 273 seconds of shader/collision preparation. Persisted private caches make subsequent launches much faster (about 20–30 seconds observed). A startup status page appears before the simulator is ready. Native Sim 5.1 is read-only; caches go to private `kit-cache51`, `kit-data51`, `cache51`, `config51` and `data51` directories. No shared installation needs modification.

The launcher now runs a bounded certificate-verified WSS readiness check before attaching Isaac Sim to OpenXR. It ACKs messages, requires a video offer and closes normally. This passed on consecutive fresh starts; complete tests decoded 165 and 276 desktop video frames after startup, with the test clients disconnected afterward. A listening port or an encoder check alone is insufficient. The previous native endpoint stall recurred during this work; its root cause is still not established, and the readiness gate is not a guarantee against all later disconnects.

CloudXR uses HTTPS/WSS port 48322, signaling 49100, and media UDP 47998. All are stopped with this owned container. No router forwarding or firewall changes were made. Quest video is operator-confirmed, and actual right-controller tracking reached the simulator. Desktop media decoding was also verified on the LAN.

## Source and dependency recovery

The full preserved source archive is `recovered/macbook-2026-10-28/robotis-lab-source.tar.gz`, SHA-256 `f5bd2f0dde369c1070ec853bd9bfc77f186f45b1fd342bee617f591d5bc1c218`. It came from the MacBook's June 30 ROG backup beneath `/Users/kadirumacbook/Documents/aiworker/backups/rog-pairog2-20260630_180523/home/robotis_ws/robotis_lab`. The original source and USD were untouched. BG2 USD SHA-256 remains `82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d`.

`runtime/prepare_quest_source.py` prepares only the extracted working copy: removes AppleDouble metadata, narrows package imports to avoid unrelated DDS/mimic/fluid dependencies, and guards malformed controller packets. Original edited files retain `.recovered` copies. The archive remains authoritative.

Private Sim-5.1 Python dependencies in `/workspace/pydeps51` were installed with `--target --no-deps --no-build-isolation`: `gymnasium==1.2.0`, `h5py==3.13.0`, `hidapi==0.14.0.post2`, `prettytable==3.3.0`, `flatdict==4.0.1`, `farama-notifications==0.0.4`, `einops==0.8.1`, `cloudpickle==3.1.1`. Sim's bundled NumPy/SciPy/Torch remain unchanged. `/workspace/venvs/isaacteleop` independently contains `isaacteleop[cloudxr]==1.0.193`, installed with NVIDIA's package index. No humanoid environment was reused. See [dependency install evidence](../evidence/quest-dependency-install.log).

Reference: [Isaac Lab CloudXR connection guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/how-to/cloudxr_teleoperation.html), [NVIDIA PhysX particle documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/particles/particles.html).

Next: rehearse small right-arm/gripper movements and trigger release/reacquisition, then attempt a controlled cup grasp/tilt and measure source/target/spill outcomes. Headset latency and grasp/contact behavior remain unverified. There is no successful pouring demonstration yet.
