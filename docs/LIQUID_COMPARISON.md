# Five-cup liquid comparison

Prepared October 29, 2026, America/Los_Angeles. This is a simulation comparison, not calibrated fluid rheology or a learned pouring policy.

## Setup

All five cups use the recovered open source-cup geometry at scene scale `(0.58, 0.58, 0.75)`: body diameter 60.32 mm, height 120 mm. Original USD assets are unchanged. Each fill has approximately 62 mL of nominal particle volume at density 1,000 kg/m³. This is a mass/spacing convention, not a measured physical volume.

| Cup | Color | Particle spacing | Fluid rest offset | PBD viscosity | Particles |
|---|---|---:|---:|---:|---:|
| 1 · Fine | Blue | 2.8 mm | 1.4 mm | 0.01 | 2,824 |
| 2 · Reference | White | 3.6 mm | 1.8 mm | 0.01 | 1,329 |
| 3 · Coarse | Orange | 4.4 mm | 2.2 mm | 0.01 | 728 |
| 4 · Low viscosity | Cyan | 3.6 mm | 1.8 mm | 0.001 | 1,329 |
| 5 · High viscosity | Purple | 3.6 mm | 1.8 mm | 100 | 1,329 |

The installed PhysX schema defines viscosity as a dimensionless coefficient. These labels do not identify real water, oil or syrup. Cups 1–3 isolate resolution at fixed viscosity; cups 2, 4 and 5 isolate viscosity at fixed resolution. The fluid rest offset is a solver parameter, not the radius of a visible ball. Individual particle spheres remain hidden.

Common physics: 200 Hz, 16 particle solver iterations, CCD enabled, max velocity 3 m/s, max depenetration velocity 1 m/s, cohesion 0, surface tension 0, friction 0.08. These are uncalibrated stability/appearance choices. No evaporation, particle expiration or deletion is implemented.

Each liquid uses a separate particle system so its material can differ. Different systems do not collide or physically mix, an [explicit PhysX limitation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/particles/particles.html). Compare one spill at a time.

## Why the earlier spill seemed to disappear

The user's saved 158-sample episode retained 1,167 of 1,331 particles on the tabletop in its last sample; 156 were past the table edge. No sample placed a particle below the slab while inside its footprint. The original recording did not include mesh geometry, so it does not alone prove the visual cause. [Recorded-position analysis](../evidence/previous-spill-diagnosis.json).

Subsequent [controlled tests](../evidence/five-liquid-filter-failure.json) reproduced a rendering failure: with one surface configuration, 1,742 fine-fluid particles remained on the tabletop while the generated mesh had no vertices there. Physical particles and the visible surface must therefore be measured independently. A point below table height after rolling back underneath from the floor is not evidence of passing through the slab; diagnostics separately track whether it previously crossed an edge and any downward crossings away from edges.

The remedy combines enough surface-grid capacity, a filter that preserves thin sheets, enabled translucency/reflections, and modest contact/flow tuning. Increasing viscosity alone cannot fix missing geometry. Rendering post-processing does not change the underlying physical particles; see [NVIDIA's particle rendering documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/particles/particles.html#post-processing-for-fluid-rendering).

## Surface and recording

The native PhysX isosurface uses grid spacing `0.75 × particle spacing`, surface distance `0.9 × spacing`, `GS` (grow, smooth) filtering, smoothing radius `0.75 × spacing`, two mesh-smoothing passes and four normal-smoothing passes. Each system is bounded to 131,072 vertices, 262,144 triangles and 4,096 sparse subgrids for the fine liquid and 2,048 for each other liquid. The thin-film appearance is an approximation at millimeter resolution; do not interpret surface mesh volume as liquid mass.

The final display uses `UsdPreviewSurface` with opacity 0.14, IOR 1.333, roughness 0.06 and a faint blue color. It is an explicit transparent inspection approximation, not a validated optical water model. The native clear-water MDL appeared dark or nearly invisible in this headless XR setup; it is preserved in the source history but is not the active material. Reflections and translucency are enabled. A noncolliding checker pattern and soft reflection light make the spill easier to inspect. The surface is rebuilt at the configured 20 Hz action/render rate while collision solving remains at 200 Hz. Physical particle smoothing remains disabled, so recorded `positions` are unsmoothed physical data.

Recordings concatenate the five fixed particle arrays into `[frames, 7539, 3]`. Episode attribute `fluid_conditions_json` records each condition and its half-open particle slice. `fluid_surface_profile` identifies the renderer. Operator saves remain outcome-unverified; synthetic arm/control tests remain explicitly marked `synthetic_validation`.

## Run the comparison

Open [the control page](http://192.168.68.60:8212/) and reload it. **View all cups** shows the color order and **View spill area** shows the comparison patch. Quest controller instructions are in [QUEST_RESTORE.md](QUEST_RESTORE.md).

Each **Spill 1–5** button resets the five cups, lets the fills settle for one simulated second, then releases that liquid's existing particles with zero velocity 8 cm above the same tabletop location. No particles are created or removed. This is a standardized drop test, not a robot-operated pour. Robot commands are held during the test. **Reset scene** restores all five fills and normal controller motion. Spills can leave through the table edge; they should not vanish while retained on the table.

The host validation uses the single existing simulator and aborts if recording or tracked controllers appear:

```sh
ssh kfuUbuntu 'python3 /home/kadir/aiworker-pouring-4070/src/validate_five_liquids.py --ids 1,2,3,4,5 --seconds 6 --prefix five-liquid-final'
```

Do not run this during an operator session. Physics time advances slower than wall time under the preserved shared-host limits.

## Measured results

In the final single-run, six-simulated-second standardized drop comparison, all five cases conserved their particle counts, had zero detected downward crossings through the interior of the tabletop, and had surface vertices near 100% of particles still on the table. Particles below table height had previously passed an edge; some later moved back under the table footprint on the ground.

| Cup | Particles retained on table | Retained fraction |
|---|---:|---:|
| Blue · Fine | 2,287 / 2,824 | 81.0% |
| White · Reference | 1,150 / 1,329 | 86.5% |
| Orange · Coarse | 656 / 728 | 90.1% |
| Cyan · Low viscosity | 1,116 / 1,329 | 84.0% |
| Purple · High viscosity | 1,011 / 1,329 | 76.1% |

**Recommended starting point: white cup, 3.6 mm spacing and viscosity 0.01.** It balances retained spill and resolution. Coarser fluid retained more in this test but creates larger visible droplets; finer fluid costs more. The very high viscosity did not improve retention, so increasing it alone is not a reliable remedy. These are one-run simulated drops, not repeated robot-pouring trials or a calibrated viscosity study. The extreme purple preset can shed a particle during initial settling.

[Full comparison evidence](../evidence/five-liquid-validation.json). The common display material was refined during this sweep without changing physics; the final transparent reference view is recorded separately. Mesh coverage is a geometric proximity check, not a claim of photorealism. Isolated droplets can still look rounded at this resolution.

Measured action throughput was 2.10–2.38 steps per wall second during comparison, and 2.39 during the 160-frame recording check. Five simultaneous fluid systems are slower than the previous single-cup scene; the configured 200 Hz physics and 20 Hz surface rates refer to simulated time. The four-core CPU quota and 4070-only GPU allocation remain intact.

Final checks also passed for [recording](../evidence/five-liquid-recording-validation.json), [five-cup metadata](../evidence/five-liquid-recording-schema.json), [pause and isolation](../evidence/five-liquid-runtime-verification.json), and [desktop streaming](../evidence/five-liquid-final-state.json). The desktop client decoded 512 frames and disconnected; Quest retesting is still operator-side. The scene is reset and not recording.
