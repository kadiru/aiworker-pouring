# Isolated AIWorker runtime

**Current profile:** the Quest/pouring setup runs with `--profile quest --fluid`, using the read-only native Isaac Sim 5.1 installation, recovered Lab 2.3, original robot collisions, three cameras and a [five-cup liquid comparison](../docs/LIQUID_COMPARISON.md). See [the Quest runbook](../docs/QUEST_RESTORE.md) for connection, controls, validation, data and identity-checked restart commands. Quest video and right-only controller input are operator/runtime confirmed; a successful grasp or pour remains unverified.

## Earlier Sim 6 baseline (historical fallback)

The remainder describes `--profile baseline`; it is not the currently running Quest profile. Use the current runbook for active container identity and restart instructions.

This is a standalone dry scene built from the recovered FFW-BG2 USD and empty cups. It does not restore the complete historical Isaac Lab task, Quest teleoperation, learned policies, or liquid simulation. Current verification status is in `../STATUS.md`.

The Ubuntu runtime lives at `/home/kadir/aiworker-pouring-4070`. Its only container is `aiworker_pouring_4070`, using the already installed `nvcr.io/nvidia/isaac-sim:6.0.1` image. The old `isaacsim6_home` container and humanoid installations are untouched.

## Resource ownership

Only RTX 4070 UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6` is exposed. The RTX 4090 is excluded. Normal CPU quota is four cores, affinity 16–23, low shares and nice priority; RAM is limited to 16 GiB with no extra swap. Temporary eight-core warm-up windows were used while the humanoid GPU was idle, with automatic return to four. These constrain resource use but are not a guarantee of zero shared-host overhead.

`launch_container.py` refuses launch if another compute process is on the 4070. It starts `gpu_priority_guard.py`, which polls every two seconds and stops only its immutable AIWorker container ID when another compute process appears on that GPU. Three consecutive inspection failures also cause AIWorker to yield. Detection and container shutdown can take several seconds. It never stops the competing job and never automatically restarts AIWorker. Five mocked guard tests cover ownership, the other GPU, foreign processes, container replacement and stale exited-process records; they do not launch competing GPU work.

First-use NVIDIA shader compilation took several minutes under the CPU quota. A native stack showed active GPU compiler work. A launch with 32 workers exhausted the original 12 GiB limit; the runtime now uses eight workers and a 16 GiB limit. The complete Kit cache is persisted at `kit-cache/` and mounted at `/isaac-sim/kit/cache`; mounting only `/root/.cache` did not retain those shaders. Debugger installation was confined to the discarded diagnostic container.

The user authorized coordination with the humanoid chat. On October 28 it confirmed that the 4070 can remain with AIWorker for now; future humanoid work should use the 4090 where feasible. Humanoid priority remains in force.

## Scene differences

- Original USD SHA-256: `82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d`. The source asset is unmodified.
- Fixed world root and 25 movable joints are inspected. The extra world fixed joint is disabled in the scene layer to avoid duplicate constraints.
- Sixteen dense convex-decomposition collision meshes use convex hulls in this dry profile. Some source meshes contain almost 280,000 vertices. Do not use this profile to validate grasp/contact fidelity.
- Gravity is disabled on robot bodies as in the recovered config; joint gains and zero pose follow that config. Physics uses CPU PhysX, rendering uses the 4070.
- A simple new table and camera placement were created for viewing. They are not a recovered calibrated task setup.
- The startup check commands head_joint1 to 0.08 radians and back to zero. The saved check measures actual simulated joint positions. All motion is scripted.

## Operation

The verified local-network viewer is [AIWorker live preview](http://192.168.68.60:8212/). It opens automatically with in-memory JPEG captures at up to five frames per second. Physics runs at 60 Hz with 30 Hz rendering. Controls request a head-motion check, pause/resume, reset robot pose, and reset view. Pose reset does not reset the cup scene. The image preview has no mouse-driven 3D navigation or Quest teleoperation.

Native WebRTC reached signaling but the server rejected video creation with `StreamSdkException 800b0000`. It is disabled in the normal runtime. The optional source branch `AIWORKER_WEBRTC=1` and `viewer-build/webrtc-client.js` remain diagnostic references, not a verified connection path. The working preview does not require host-driver changes, NVENC or a native streaming client.

Browser controls and visible robot/cup rendering were verified on October 28. Pause held simulated time at 58.966669742 seconds while image timestamps kept advancing; resume advanced time again. A browser-triggered head check measured a 0.080149613 rad peak for a 0.08 rad command and returned within 2.58e-9 rad of zero. Evidence: `../evidence/browser-control-check.json`, `../evidence/joint-motion-check.json`, `../evidence/runtime-verification.json`, and `../evidence/aiworker-live-view.png`.

Observed steady state: about 3.76 GiB host RAM, 1.4 GiB GPU memory attributed to Isaac Sim, and about 1.5 CPU cores. These are samples, not guaranteed maxima. The GPU-0-bound original container remained stopped with its September 5 start timestamp unchanged.

Run commands on Ubuntu:

```sh
docker logs --tail 40 aiworker_pouring_4070
cat /home/kadir/aiworker-pouring-4070/outputs/status.json
cat /home/kadir/aiworker-pouring-4070/outputs/joint-motion-check.json
docker stop --timeout 10 aiworker_pouring_4070
```

To relaunch after an intentional stop, inspect the container and guard log first. Remove only this stopped, project-owned container, then use the guarded launcher. Do not use `docker start`, which bypasses its GPU preflight and guard attachment.

```sh
docker inspect --format '{{.State.Status}} {{index .Config.Labels "owner"}}' aiworker_pouring_4070
docker rm aiworker_pouring_4070
python3 /home/kadir/aiworker-pouring-4070/launch_container.py
```

The HTTP viewer binds `192.168.68.60:8212` on the private LAN. No router/firewall/public exposure was configured. The disabled native WebRTC branch is configured for TCP 49110 and UDP 48008.

## Sources

- [Isaac Sim 6.0.1 livestream setup](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/manual_livestream_clients.html)
- [NVIDIA web viewer SDK](https://docs.omniverse.nvidia.com/ov-web-sdk/latest/web-sample/overview.html)

The active preview uses ordinary browser image requests and small JavaScript controls. The active viewer uses the plain `viewer-build/index.html` and `main.js`, copied to the remote `viewer/` directory. The earlier Vite output and lockfile are retained for history. The retained WebRTC diagnostic client uses `@nvidia/ov-web-rtc` 6.2.2 from NVIDIA's official package registry.
