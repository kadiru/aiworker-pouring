# Fountain parameter search

This is a measured Isaac Sim experiment for finding a useful water/wine-like PBD preset. It is not a calibrated CFD model, a globally exhaustive optimization, or a learned visual water classifier.

## Scenario

The standalone `fountain` profile uses the existing guarded AIWorker RTX 4070 container. It replaces that container's robot/Quest scene while the experiment runs; the original Quest launch profile and assets remain available. It never starts a second simulator or changes humanoid processes.

The receiver has a 100 × 80 mm interior, 75 mm walls, and a solid 12 mm floor. A nominal 10 mm nozzle pours from 200 mm above the floor at 25 mL/s for three simulated seconds. The nozzle begins at the center, moves 25 mm sideways between 1.25 and 1.75 seconds, then stays off center. About 75 mL accumulates. Flow stops, followed by three seconds of passive settling.

For the additional sloshing test, a known **synthetic** +0.08 m/s horizontal velocity impulse is applied to the pool at 3.5 seconds. The trial continues to eight seconds. This is a controlled fluid-state perturbation, not motion commanded through a physical container or a robot. It separates wave response from incoming jet impact.

Particles are emitted at physics-step cadence. The emitter maintains per-particle mass as the set grows, flushes physical USD readback before appending, and accounts for every emitted layer. It never expires or deletes particles. Nominal volume is `count × spacing³`, consistent with the chosen rest-density convention; it is not a measured physical pour volume. An 80,000-particle hard cap (for the bounded 1.0 mm convergence trial) bounds each trial.

## What is measured

- Finite positions/velocities, expected particle count, emitted nominal volume, retained volume, escape outside the receiver, and interior floor penetration beyond a 1 mm tolerance. A penetration into the slab is not necessarily a particle passing all the way through it.
- Jet occupancy along height, cross-section variation and deviation from gravity-driven free fall.
- Pool area occupied, spatial variation of surface height, mound persistence, velocity RMS during pouring and after flow stops.
- Particle-neighbor counts and proximity to the container are **geometric proxies**, not reported solver contact impulses or collision forces.
- Mesh vertices near physical particles, independently of physical retention. Transparent rendering must not hide missing surface reconstruction.
- Center-of-mass motion after the known impulse. A bounded damped-sinusoid fit estimates oscillation frequency, damping and fit quality. The ideal first-mode rectangular-basin frequency is only an approximate inviscid sanity check.

The first score combines retention (20%), jet continuity (15%), free fall (10%), spreading (15%), leveling (15%), settling (15%) and motion during pouring (10%). It can favor an excessively damped liquid; the high-viscosity control demonstrated that weakness. The second score gives equal weight to the first score and the measured sloshing response. Its weights and acceptance bounds are engineering hypotheses, not a waterness probability. Poor retention, nonfinite data, count changes, penetration or unstable speeds near the apparatus disqualify a candidate. Escaped droplets falling below the bench are excluded from the speed gate and remain counted as lost volume. Early global-speed warnings included that legitimate free fall; finalists are rerun with the corrected local gate. A weak wave response additionally disqualifies a sloshing candidate.

Physical parameter sweeps and appearance comparisons are separate. Native PBD viscosity/cohesion values are not SI material measurements. Water and wine display buttons change color only.

## Search strategy

1. Eighteen coarse conditions cover particle spacing (1.8–3.6 mm), viscosity, cohesion, surface tension, friction, adhesion, global damping and vorticity, including a high-viscosity control and a deliberately noncolliding receiver.
2. Shortlisted settings undergo the pool impulse test. Fine-particle cases test 400 and 800 Hz solving to expose numerical sensitivity.
3. A bounded follow-up varies explicit receiver friction, interaction radius/neighborhood, stronger cohesion, vorticity and fill depth. This follows the observed overly damped responses; it is not an unbounded Cartesian search.
4. Local refinement, repeated lattice orientation, smaller time step, and changed flow/height test a promising candidate.
5. Surface comparisons hold physics fixed while varying mesh reconstruction, opacity/roughness and the native clear-water MDL. Rendered sequences receive separate visual inspection. No automatic external VLM has been run.

The first diagnostic prototype emitted only on render frames, producing a nozzle gap. It is excluded from the search. Subsequent runs use physics-rate emission. A later defensive readback audit measured zero position/velocity correction across 423 baseline emissions; stale USD readback was not established as the cause of the observed damping.

## Operation and artifacts

The [live experiment](http://192.168.68.60:8212/) provides a camera view, measurements, ranked candidates, bounded search commands and replay. Pause holds the physical state. Stop clears queued trials and pauses the current one. “Replay best tested” uses a separately selected, completed eligible trial when available; otherwise it replays the current best screening result.

Source: `runtime/src/run_fountain.py`, `fountain_plan.py`, `fountain_metrics.py`, `fountain_report.py`. Six CPU-only checks verify failure gates and the sinusoid fit against known synthetic data; these checks do not establish simulator realism. The unchanged GPU priority guard has five additional boundary tests.

Remote artifacts are under `/home/kadir/aiworker-pouring-4070/outputs/fountain/`. Each trial records parameters, per-frame measurements, compressed physical trajectories, rendered frames and an impulse record where applicable. `latest.json` retains prior completed results across restarts. `best-tested.json` contains the best screening result; `selected-preset.json` records a review-based choice by completed trial path. Run directories preserve the provenance of earlier stages.

To replace the owned container, first inspect its immutable ID and owner label, then use:

```sh
python3 /home/kadir/aiworker-pouring-4070/restart_owned.py \
  --expected-id <verified-current-ID> --profile fountain
```

To restore the five-cup Quest profile, use the same identity check with `--profile quest --fluid`. Both profiles retain the four-core quota, low CPU priority, 16 GiB memory cap, one-GPU visibility and priority guard. The launcher refuses an already occupied 4070. Never stop or modify another container or GPU job to make room.

## Completed experiment — November 1, 2026

Fifty recorded trials cover coarse conditions and controls, sloshing, readback diagnostics, targeted follow-ups, resolution, independent validation, a diagnostic replay and six appearance comparisons. One final replay brings the saved result count to **51**. The earlier render-rate-emission prototype is excluded. This is a bounded search with follow-ups, not an exhaustive Cartesian search through every possible parameter/value.

The reviewed practical selection is **`render_finer_grid`**, using the physical settings of `resolution_1p8`. [Exported parameter record](../evidence/fountain/water-preset.json) includes the full specification, derived offsets, fixed scene, validation records and limitations. [Selection record](../evidence/fountain/selected-preset.json) identifies the exact completed source trial. The highest numerical heuristic score remains separately saved as `best-tested.json`; that 1.0 mm trial costs substantially more for little score improvement.

| Setting | Selected value |
|---|---:|
| Physical particle spacing | 1.8 mm |
| Physics step / position iterations / neighborhood cap | 2.5 ms / 16 / 96 |
| Added viscosity, cohesion, tension, friction, adhesion, damping, vorticity | 0, native coefficients |
| Particle/contact offset / solid rest offset / fluid rest offset | 1.51515 / 1.5 / 0.9 mm |
| Surface grid / surface distance / smoothing radius | 0.9 / 1.35 / 0.9 mm |
| Mesh / normal smoothing passes | 2 / 4 |
| Display opacity / roughness / IOR | 0.14 / 0.06 / 1.333 |

Zero **added** viscosity does not mean real water has zero viscosity. This solver still exhibits substantial numerical damping. Surface controls change reconstruction, not the physical particle trajectories.

The ordinary six-second fountain retained **12,617 of 12,873 particles (98.01%)**, corresponding to **73.58 of 75.08 mL nominal volume**. No interior floor penetration was detected. All remaining particles are accounted for as escaped, not expired. The reconstructed surface was near all retained pool particles at the final sample. Mesh proximity is a geometric diagnostic, not proof of optical realism or mesh watertightness; deep interior particles can legitimately lie farther from the boundary mesh.

The eight-second sloshing reference took **52.96 wall seconds** at 1.8 mm. The 1.2 mm trial took 137.0 seconds; 1.0 mm took 245.36 seconds, approximately 4.6 times as long, for only a 0.52-point combined heuristic increase. Finer resolution did not resolve damping. [Measurement plots](../evidence/fountain/measurements.pdf) show the actual trajectories and fitted responses.

| Independent variation | Retained | Interior penetration detected | Main qualification |
|---|---:|---:|---|
| Different lattice orientation | 97.59% | 0 | Single repeat, not a confidence interval |
| Density-based mass assignment | 97.94% | 0 | Similar behavior to explicit particle mass |
| Half step, more iterations and neighbors | 90.10% | 0 | Numerical sensitivity; three solver settings changed together |
| 35 mL/s, 25 cm fall | 93.90% | 0 | Higher-flow operating point only |
| 15 mL/s, 15 cm fall | 99.86% | 0 | Failed weak-slosh-response gate |

The base slosh fit gives **0.825 Hz and 3.1/s decay**, versus an approximate ideal inviscid frequency of **1.48 Hz**. Some other fits hit the 4/s search bound, so their damping estimates should not be interpreted as precise material measurements. Waves disappear too quickly, and low-flow behavior fails the response check. Retention and leveling scores alone are therefore insufficient to declare the model water-like. Real-water reference footage and further numerical calibration are still needed.

[Six-way surface comparison](../evidence/fountain/surface-comparison.png) keeps physics identical. The finer grid gave modestly smaller surface lobes while preserving a visible settled pool. Round splash droplets remain. This was qualitative agent inspection, not a calibrated visual score or external VLM assessment. All six appearance trials retained the same 98.01%.

[Final fountain video](../evidence/fountain/selected-fountain.mp4) contains 60 rendered frames played at 10 fps, representing six simulation seconds. Native preview capture is asynchronous and may lag a frame. The selected appearance trial took 43.6 wall seconds under the existing caps; the separate final replay included a brief inspection pause and is not a throughput measurement. The final replay has no synthetic slosh impulse. [Raw selected trial](../evidence/fountain/selected-trial/) includes metrics, samples, 4 Hz physical trajectories and rendered frames.

The [live lab](http://192.168.68.60:8212/) is left with the completed pool held, no queued experiments and the selected preset available through **Replay best tested**. [Final runtime evidence](../evidence/fountain/final-runtime-verification.json) records the immutable container ID, running guard, 4070-only device allocation and unchanged CPU/RAM limits. [All metadata](../evidence/fountain/metadata.tar.gz), [final deployed source](../evidence/fountain/runtime-snapshot-final/) and [hash manifest](../evidence/fountain/source-sha256.json) preserve reproducibility. The five-cup Quest scene has not been retuned with this provisional preset.

## References

[NVIDIA's particle guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/particles/particles.html) documents the separation of physical particles and rendering, contact/rest offsets, mass convention, and the lack of interactions between separate particle systems. [PhysX PBD material API](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/class_px_p_b_d_material.html) documents the native coefficient interface. The installed Isaac Sim 5.1 / OmniPhysics 107.3 source is the implementation reference used by this experiment.

The inviscid frequency sanity check uses `omega² = g k tanh(k h)`, `k = pi / L`, as given in [NASA SP-106, section 2.2](https://ntrs.nasa.gov/api/citations/19670006555/downloads/19670006555.pdf). Surface tension, finite particle resolution, numerical damping and wall effects are not calibrated by that formula.
