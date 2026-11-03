# Hinged receiver with obstacles

The live water lab now has **Pour, tilt & slosh**. It fills the receiver, raises one edge, lowers it to the table, and observes water moving around two solid blocks. The older fountain-only replay remains available separately. The original “stir” label described moving the incoming nozzle, not moving the container; it is now labeled **Move stream** for that older experiment.

## Motion and geometry

| Simulation time | Action |
|---|---|
| 0–3.0 s | Pour approximately 75 mL nominal volume around the blocks |
| 3.0–3.7 s | Stop flow and let the initial splash subside |
| 3.7–4.5 s | Lift the opposite edge smoothly to 22° |
| 4.5–4.85 s | Hold the tilted container |
| 4.85–5.15 s | Lower it with an accelerating return, stopping level on the table |
| 5.15–10 s | Observe the resulting waves and settling |

The hinge axis runs along the lower left edge, at `(−55, 0, −12)` mm in world coordinates. The opposite lower edge rises **41.2 mm**. The floor, walls, checker tiles and both blocks share one rigid transform. The two blocks are fixed to the container: amber **12 × 14 × 28 mm**, blue **14 × 14 × 36 mm**, with different off-center placements.

The receiver is a **scripted kinematic body**, following a fixed hinge path. Its colliders move through the actual PhysX particle simulation. Fluid positions and velocities are not directly changed to create the slosh. This is not a gravity-driven drop, an actuator force simulation, or a measured table-impact force; the return stops at the table plane by prescription. [NVIDIA's kinematic-body documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html#kinematic-rigid-body) describes how the body and its child colliders move together under application control.

The liquid retains the previously selected provisional 1.8 mm / 400 Hz settings. Kinematic rigid-body CCD is unsupported in this engine and is not enabled. Existing particle/scene CCD and the small physics step are preserved. No other simulator, humanoid process, driver or GPU resource setting is changed.

## Measurements and checks

Retention, floor checks, surface variation, center-of-mass motion and relative velocity are measured in the **moving container frame**. This prevents container rotation itself from being mistaken for fluid motion. Raw saved particle trajectories remain in world coordinates.

The moving-floor diagnostic checks intrusion into the finite 12 mm slab and downward crossings between adjacent 20 Hz samples. A regression test distinguishes these from a droplet already far below the scene. The first diagnostic run flagged an escaped particle about 2.8 metres below the scene when rotation projected it beneath the receiver; the original result is preserved under `evidence/hinge/initial-diagnostic/`. The corrected check does not erase escaped particles from retention accounting. Sampled checks can miss events between samples and are not a proof of perfect collision handling.

Block checks record particles more than 1 mm inside either solid and the number near their surfaces. Proximity is a geometric proxy, not a native collision impulse. Center-of-mass reversals require at least 0.2 mm prominence to suppress numerical noise. These are response measurements, not a calibrated water-quality score.

**Stationary comparison** repeats the same fill, geometry, obstacles, physics and observation duration with zero hinge angle. It separates the effect of container motion from the residual pouring splash. Hinge experiments use protocol 4 and do not participate in the earlier fountain-parameter winner selection.

## Operation and provenance

Open [the live lab](http://192.168.68.60:8212/) and click **Pour, tilt & slosh**. A full ten-second simulated cycle takes roughly 70 wall seconds under the existing four-core/16 GiB limits. The phase highlight, tilt angle and edge height show progress. **Pause / resume** holds the body and water. **Replay fountain only** runs the previous nozzle-only sequence; **Stationary comparison** is the control for this new experiment.

Implementation: `runtime/src/fountain_hinge.py`, `run_fountain.py`, `fountain_hinge_report.py` and `runtime/fountain-viewer/index.html`. Eleven CPU tests cover pivot geometry, table clearance, moving-frame velocities, finite-floor crossing classification, obstacle penetration and existing fluid metrics. Actual simulator runs separately test collider interaction.

As with the earlier preset, waves are overly damped. This new fixture exposes and quantifies the deficiency; it does not establish that the liquid is physically calibrated water or wine.

## Verified result

The final ten-second run retained **12,636 of 12,873 particles (98.159%)**, with **zero detected finite-floor intrusions/crossings and zero block penetrations**. After returning level, the pool center traveled **21.67 mm** and reversed direction **twice** above the 0.2 mm prominence threshold. Peak post-return RMS speed was **0.1695 m/s**, versus **0.01028 m/s** in the stationary control (16.49 times higher). This shows a response to moving colliders rather than just the preceding fountain splash. The water still damps quickly.

- [Full cycle, played at simulation-time speed](../evidence/hinge/hinged-container.mp4)
- [Tilt and return at half simulation speed](../evidence/hinge/tilt-return-half-speed.mp4)
- [Measured comparison plots](../evidence/hinge/response.pdf) and [comparison JSON](../evidence/hinge/comparison.json)
- [Four rendered phases](../evidence/hinge/sequence.png) and [live raised-edge screenshot](../evidence/hinge/live-tilt.png)
- [Raw trials, final sources and reports](../evidence/hinge/hinge-evidence.tar.gz), [tests](../evidence/hinge/metric-tests.txt), [runtime/guard verification](../evidence/hinge/runtime-verification.json)

The first uninterrupted moving run took 71.45 wall seconds. The final recording includes a pause for visual inspection; that pause is excluded from the simulated-time movie and makes its wall-time field unsuitable for throughput comparison. Render captures can lag a frame. Both trials preserve raw world-space trajectories; measurement coordinate transforms do not move the physical particles.
