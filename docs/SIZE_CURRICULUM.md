# Cup-size curriculum and bounded recovery

Started November 8, 2026. Separate native study:
`/workspace/learning/A-size-20261108-v1`, prefix `SZ_A_`.
The completed V24 benchmark and its 25/30 result remain unchanged.

## This stage

Nine stationary camera/containment pilots cross internal cup-height scales
0.9, 1.0, 1.1 with diameter scales 0.9, 1.0, 1.1. The 4 mm base and 2.4 mm
wall thickness stay fixed. Inner rim height is 34.6–41.4 mm and inner rim
diameter is 43.2–52.8 mm. Bottle geometry, material A, liquid coefficients,
particle spacing and physics timestep stay fixed.

The cup mesh, convex wall colliders, particle containment mask, measured rim
height and analytic geometric capacity use one profile. Capacity scales as
height × diameter². Sparse initial cup inventory uses the same ratio, with a
baseline of 16 **nominal** mL. That inventory reference is not a measured
physical capacity or a full-level calibration. The stationary pilots are
low-fill containment/camera checks, not pouring or grasp demonstrations.

The first grid-initialized round closed **8/9 contained**. Tall/narrow case 06
lost one particle (0.005832 nominal mL): outside by 0.105 s, near the worktop
by 0.405 s. Source inventory stayed unchanged. The precise mechanism is not
established because this first round lacks per-particle trajectories. All
nine original outcomes and hashes remain in `grid-stage-results.json`.

A separate nine-case revision selects the required number of particles from
the lowest available layer of the previously validated native settled seed,
clipped to each cup's interior with wall clearance. Coordinates and spacing
are not rescaled. The revision changes initial arrangement only; liquid and
collision settings remain unchanged. It records sparse outside-particle
positions/origins for diagnosis. This tests a better initialization; it does
not establish leak-free dynamic pouring or explain the original loss.

Camera crop normalization accepts declared cup geometry. These dimensions
are known fixture configuration, not camera-inferred size. Resized autonomous
episodes are rejected unless a checkpoint carries the matching declared
geometry and explicit native acceptance. Size-aware template localization,
mouth clearance, mass-flow conversion and full-level prediction still need
calibration and validation before that acceptance can be granted.

## Recovery supervisor

V24 learned weights are retained. A programmed supervisor adds:

- Early tilted views rely on the already verified visual lift plus measured
  gripper obstruction; the legacy visual absence gate becomes active at 75°.
  Missing encoder obstruction still causes recovery after 0.35 seconds.
- Sustained high wrist-empty probability with ambiguous source presence
  causes `source_state_unconfirmed` after 1.2 seconds. This is an abstention
  and a failed task; it cannot count as a successful empty detection.
- Eight seconds at the tipping cap without a normal stop, or 48 seconds
  after active pour control begins, causes `pour_progress_unconfirmed` and
  return. The original successful full/empty thresholds remain unchanged.

All thirty **former** V24 test traces are explicitly consumed for this
follow-up's development replay. The original 40-second proposed overall
timeout would interrupt two formerly successful traces; 48 seconds does not
interrupt any of the 25 successful traces before their original stop. Case
13 requests uncertainty recovery at 26.955 seconds in the old trace. Replay
establishes only the first changed decision, not future contact, spill,
return or task success. No new model fitting has occurred.

Six native development regressions reuse old cases 13, 23, 15, 19, 00 and 29.
These test the two source failures, two unresolved full-stop failures, and
one successful full/empty control each. They are deliberately **not fresh
held-out tests**. Every outcome remains in `stage-results.json`. No failure
is relabeled as success because a safety return finishes.

## Acceptance and next stages

Stationary pilot acceptance: constant commanded joints, tail arm movement
below 0.3°, tail level standard deviation below 0.01, tail inventory range
below 0.05 nominal mL, and zero final-outside/ever-near-worktop inventory.
Native HDF5, camera clocks, transitions and source hashes are audited.
This checks low-fill containment only, not near-full or dynamic pouring.

Full task acceptance retains the prior strict rules: physical and visual
grasp; bilateral contact in at least 80% of tipping samples; meaningful
transfer; correct full/empty stop; ordered upright/place/release/retreat;
settled height at least 90% rim or bottle remaining at most the declared
empty allowance; zero measured loss; no fallback or unresolved gates.

Before training across sizes, repair independent source-presence confirmation
and the full-level predictor/agreement failures, and validate settled labels
near the rim. Then collect fixed-bottle demonstrations across cup sizes,
positions and starting fills, fit the supervised perception/BC components,
and evaluate on reserved sizes and combinations. Bottle variation follows
after rechecking grasp width, neck/lip clearance, reach, mass and inertia.
Lighting/material B/C variation follows geometry so failures remain
diagnosable. There is no claim of physical-robot transfer.

The current method remains hybrid supervised perception and motion
imitation with IK and programmed stopping/recovery, not reinforcement
learning. The size goal is to estimate normalized settled level `h/H` and
stop with `h_final/H >= 0.90`, while escaped particle count is zero and the
return/release completes. Size randomness alone does not teach those rules.

## Commands and artifacts

Local checks:

```sh
PYTHONPATH=runtime/src .venv/bin/python -m unittest test_pour_size -v
.venv/bin/python tools/replay_size_watchdog.py
```

Native audit inside the owned container:

```sh
/isaac-sim/python.sh /workspace/learning/A-size-20261108-v1/check_size_stage.py
```

`tools/prepare_size_stage.py` prepares a separate candidate and plan.
`tools/prepare_size_seed.py` archives the original grid round and prepares
the independently retained settled-seed revision. Forty-four CPU checks
pass after adding this initializer.
`tools/continue_size_stage.py` is a finite coordinator: wait for nine
stationary pilots, require containment, archive the first candidate, apply
the replay-checked timeout, restart only the idle owned container, run six
development regressions, audit, and stop. A failed guard, audit, runtime
identity check or deadline stops the coordinator. It is not an automation
and does not restart a container stopped by the GPU priority guard.
Its first run stopped at the 8/9 containment gate, before any autonomous
attempt. `--seed-calibration` gates the next regression queue on all nine
revised pilots; it cannot erase the original failed round.

The 4070-only binding, four-core quota, CPU set 16–23, shares 256, 16 GiB
memory/swap cap and foreign-job priority guard are preserved. Point-in-time
snapshots cannot establish zero shared-host overhead. Recordings, model
weights and images stay outside Git; source, small reports and hashes are
versioned. Completed V24 parent files are hash-checked for immutability.
