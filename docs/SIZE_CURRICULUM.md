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

The revised nine pilots all pass low-fill containment and recording checks.
A frozen-weight CPU perception probe uses image-localized cup XY and known
declared dimensions, with fixture positions used only for scoring. With the
actual temporal confirmation rule, **4/9 cups confirm and 5/9 abstain**.
Among the four confirmations, maximum localization error is 4.463 mm and
settled-level mean absolute error is 4.982 percentage points (maximum 10.742
points). These are low-fill views at one consumed layout, not near-full or
autonomous success rates. The simpler single-frame probe is retained
separately; its larger errors do not describe the controller's temporal gate.
No weights were fitted by either probe. Cup-size perception needs supervised
calibration before resized autonomous trials.

Recorded camera intrinsics and extrinsics match the fixed model calibration
in all nine settled-seed pilots. A separate frozen-detector diagnostic finds
that all five rejected cups fail to reach the unchanged 0.46803 anchor-score
threshold anywhere in the two-second confirmation window. Two also jump
between widely separated image positions. This describes the rejection;
it does not prove the underlying visual cause. New size-specific localization
examples are needed; weakening the confidence gate is not validated by these
observations. See `localization-diagnostics.json`.

An ideal projection check using the fixed camera and all fifteen declared
pilot/regression fixtures gives 24.438–32.794 pixels of usable cup height in
the 384×384 task view. A one-percentage-point near-full height change projects
to only 0.249–0.336 pixels. This is a calculated image scale, not a measured
perception-error bound. Validate a closer or higher-resolution cup view before
broad size training; upsampling the existing normalized crop adds no new
captured detail. `camera-resolution.json` retains the calculation and source.

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

The six attempts are complete: **3/6 strict passes, 5/6 zero measured loss,
6/6 ordered return/release sequences**. Every attempt ran once with unchanged
weights, runtime sources and controls. The original full-size cup is used here;
these are consumed-layout development comparisons, not resized pours or a
new held-out benchmark.

| Case | Intended stop | Strict result | Final rim height | Source remaining, nominal mL | Reason |
|---|---|---|---:|---:|---|
| 13 | Empty | Fail | 57.8732% | 0.000000 | Uncertain source; two-particle spill |
| 23 | Full | Pass | 94.2206% | 33.195744 | Correct full stop |
| 15 | Full | Fail | 96.5172% | 27.252936 | Full stop unconfirmed |
| 19 | Full | Fail | 89.5465% | 40.246632 | Full stop unconfirmed; final sample under 90% |
| 00 | Full | Pass | 93.6038% | 27.777816 | Correct full stop |
| 29 | Empty | Pass | 52.6313% | 0.000000 | Correct empty stop |

Case 13 requests `source_state_unconfirmed` at 26.655 s and completes its
return at 34.405 s. Two particles first reach the worktop at 25.555 s,
1.10 s before stopping. The physical loss mechanism is unproven; completion
of the uncertainty recovery cannot count as successful empty detection.
Case 23, formerly an early source-tracking abort, now completes a full pour.

Case 15 still exhausts thirteen pauses: its six-frame prediction quartile
is 91.4294%, below the 92.5414% confirmation target, while the final measured
level is 96.5172%. Case 19 also stops `fill_unconfirmed` after thirteen
pauses. Its final-second median is 91.7190%, while its final sample is
89.5465%. The additional temporal diagnostic does not replace the original
failed score. Near-full label stability and a predeclared settled evaluation
window need review before subsequent training/evaluation.

All six returns have zero sampled stationary upright over-cup dwell and no
post-release gripper contacts; terminal bottles are upright on the table
and fingers are open. These are 20 Hz scalar observations plus selected
frame review, not a claim about unsampled contacts or physical hardware.

Native checks cover **24 recordings / 6,349 transitions**, unchanged parent
files/learned weights, and fixed camera calibration. All **102 scalar
sidecars** are SHA-verified locally. All **72 canonical selected native
frames** are inspected; this is not continuous manual playback. The simulator
is idle with an empty queue. `stage-summary.json` and `recovery-results.csv`
retain all six independently recomputed scores; `scalar-backup.json` records
the local backup. Original HDF5 files stay on Ubuntu.

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

Before training across sizes, validate a closer or higher-resolution cup
view, repair independent source-presence confirmation and the full-level
predictor, and validate stable settled labels near the rim. Then collect
fixed-bottle demonstrations across cup sizes,
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

Final local closure (after copying the completed native audit/review artifacts):

```sh
.venv/bin/python tools/backup_size_metadata.py
.venv/bin/python tools/summarize_size_stage.py
```

The backup helper requires SSH access to the owned runtime; the summarizer
checks local sidecar bytes/hashes, native/source/model identities, all six
scores, the selected-frame review and final idle/resource snapshots. It
refuses incomplete stages. Model weights, recordings and images are excluded
from Git; the recovery candidate is backed up locally as
`evidence/size-A/best.pt` (SHA recorded in `stage-summary.json`).
