# Cup perception and source-presence repair

Study: `A-perception-20261108-v1`. Started November 8, 2026. This is a
separate development study; the V24 25/30 benchmark and V25 3/6 consumed-layout
regressions remain immutable. Parent checkpoint is
`a38f447160d8f9de664494f6d685e628daa9c0d647ecf922d5223b7d892bdeaf`.

## Evidence and current interpretation

Three 768×768 camera pilots and three particle-backed repeats retain all
fluid and pass final-second label stability checks. The close view projects
about 81.5–81.8 pixels across usable cup height, versus 24.4–32.8 in the
previous task-camera projection diagnostic. Native camera clocks and
transition continuity are audited. Selected still frames were inspected;
this is not continuous manual video review.

The close lens does not cover every wider/swapped fixture. A 1024×1024
camera with a 30 mm lens is therefore prepared, with the same fixed eye and
lower cup-centered aim. The 71-fixture ideal projection screen and three native
wide-view pilots pass. A fourth native pilot verifies the revised static grip.
These final views project 72.7–84.5 pixels across usable cup height; four
three-camera composites and two full-resolution detail frames were inspected. Task/wrist cameras remain 384×384.

Two high-fill pilots retain exactly 63.504648 nominal mL, but their original
surface labels differ by 2.3841 percentage points. A separate spatial
column-top measurement preserves a 2.4107-point difference. Recomputing it
from saved native particle positions passes. Thus a fixed inventory does
not establish a unique settled surface height in this PBD fixture. The
physical/numerical cause is not established. The discrepancy must not be
removed by assigning both images an inventory-derived target.

Training labels instead use each recording's final-second median surface
measurement. Required: at least 20 samples spanning approximately one
second; height IQR ≤1 percentage point; half-window drift ≤0.5 points;
inventory range at most one nominal particle; no final outside fluid or
ever-near-table loss. Raw observations and original final-sample task
scores remain unchanged. The column-top diagnostic is additional evidence,
not a replacement success score or a runtime policy input.

Four native empty-hand fixtures are separately assigned to presence
training/validation. The earlier 32 wrist-reference episodes contain no
eligible absent-object validation examples under the projected-envelope
label. The first fitting preparation therefore stopped before optimization.
The new fixtures address that data gap; they are authored calibration,
not autonomous grasp successes. Held-bottle static fixtures explicitly use
a kinematic bottle; autonomous episodes retain the dynamic bottle. Initial
finger positions now come from 272 observed bilateral-contact frames in a
consumed recovery episode: [38.1420, 45.9844, 38.4342, 45.3812] degrees,
with the original 45-degree drive targets. The corrected native pilot ends
at [40.0999, 52.4642, 39.9889, 54.9947] degrees with bilateral contact and no
loss. Earlier authored open-finger held poses are diagnostic only. All 68
planned collection, holdout and autonomous geometry checks pass; native
fluid and autonomous success still require their respective evaluations.

The separate presence fit now passes its component gate: 48,001 parameters,
1,895 training / 659 validation images from 26 / 10 episodes, 80 epochs and
1,200 optimizer steps, 11.035 seconds on the RTX 4070. Selected epoch 40 has
zero false-positive and zero missed-positive validation frames. The 28
negative validation frames come from only two native authored empty-hand
episodes. This is a limited component validation, not a task-success rate.
The original empty detector and active controller remain unchanged at this
stage. The fitted component SHA is
`f61fe036112778ea16499778b3091731a00bbda6860b5a31d811f9be6138b2f0`.

## Completed collection and rejected localizers

All 47 cup scenes are complete: 27 training / 20 validation. Together with
14 preparation records, all 61 pass native recording and stable-label audits,
with zero measured loss. All 244 scalar sidecars are SHA-verified locally.
Four contact sheets cover all 47 selected final cup views; they were inspected.
The detail view projects 60.9–94.2 pixels across usable height. Actual stable
levels range 27.93–97.67%; near-full baseline validation is 95.09% and 97.67%.
The finite reused seed does not provide near-full coverage for all wider cups.

The first 24 gradient-template localizers are rejected: best maximum error
6.089 mm and negative confidence separation, versus the declared 3 mm and
positive absent-mask separation requirements. This 28.589-second 4070 run
stopped before cup-level optimization. A separate geometric rim probe also
fails: some early frames lack the fully rendered liquid surface, and stronger
reflections/lower glass edges distract matching. Later-frame constant optical
depth calibration improves detection but still has 16.002 mm maximum error.
All candidates remain preserved; no validation gate was relaxed.

A supervised fully convolutional rim-center heatmap with subpixel offsets
was fitted in two retained revisions. The first (33,699 parameters) failed;
the wider-context version (42,947 parameters, 108.711 seconds fitting) still
fails all-size validation at 6.782 mm maximum for its best all-size candidate.
Its original-cup views pass: 2.476 mm maximum validation / 2.566 mm training
error. This narrower scope was declared before any reserved evaluation.
An additional fit will include twelve new low/near-full baseline scenes at
six existing training layouts and four explicitly reused camera/grip pilots.
The twelve new geometry screens pass; native collection is ongoing.
Recorded splits stay unchanged, and the fit manifest records pilot reuse. Its targets are offline projected physical rim centers;
runtime input is RGB. The loss is focal heatmap loss plus twice smooth-L1
offset loss. Training adds integer image translations, mild photometric
changes and explicitly synthetic absent-cup masks. Old train/validation task
views are replayed alongside the new detail views. No reserved images or
autonomous outcomes have been consumed for this repair.

## Prepared learning and evaluation

The method is supervised perception plus the existing contextual motion
behavioral cloning/IK and programmed grasp/stop/return sequence. It is not
reinforcement learning or end-to-end motor learning.

- Cup localization: supervised RGB center heatmap and subpixel offsets,
  calibrated rejection confidence and temporal consensus. Known fixture dimensions convert image
  coordinates to tabletop coordinates; size is not visually inferred.
- Level: fine-tune the cup CNN on image-localized, normalized detail-camera
  crops, minimizing weighted squared error to measured settled surface.
  An independent spatial-color-envelope ridge regression checks agreement.
  Its current 19-feature version still fails: 9.720-point maximum error
  against the unchanged 6-point gate. The CNN alone passes (1.506-point
  MAE, 3.473-point near-full maximum error), so the combined component is
  rejected until both pass. Replay
  retains original lower-resolution task-camera images from 24/8 earlier
  training/validation reference episodes. New stationary records have weight
  four; replay has weight one. Stable replay windows use measured trailing
  medians; flowing frames retain transient labels. This protects prior visual
  coverage without inventing detail-camera images or consuming reserved tests.
- Presence: separate wrist CNN, binary cross entropy, AdamW and augmentation.
  Offline bottle pose is used only to label projected presence/absence in a
  fixed wrist crop. Live control combines RGB probability with previously
  verified grasp and measured finger obstruction. Empty recognition is
  retained separately.

Active cup collection: 27 training and 20 validation scenes over nine
declared cup geometries (height/diameter independently 0.9/1.0/1.1), three
fill cuts and wider/swapped training fixtures. Nine separate perception
holdout scenes are not queued before component selection. Six development
and six reserved autonomous realizations have separate IDs and small
position jitter. Training/validation split by whole episodes and fixture
sources, never by random frames from the same recording.

Autonomous tasks retain the original strict criteria: verified grasp,
meaningful transfer, correct stop, ≥90% final rim height or empty source,
zero escaped particles, upright set-down/release and hand withdrawal.
Resized autonomous pours and bottle-size variation remain gated. Component
accuracy alone is not autonomous success.

## Isolation and reproduction

Only immutable owned container
`0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`
is operated. It retains RTX 4070 UUID
`GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`, four-core quota, CPU set 16–23,
shares 256, 16 GiB memory/swap, and the foreign-GPU-job priority guard.
Fitting starts only while simulation recording/queue is idle. No other job
is signaled or reconfigured. Snapshots do not prove zero shared-host overhead.

Native helpers live under `/workspace/learning/A-perception-20261108-v1`.
`prepare_perception_stage.py` creates the isolated plan; the geometry and
collection preparation helpers archive earlier candidates before changes.
`check_perception_stage.py` audits records and exports native stills.
`validate_perception_surface.py` recomputes labels from trajectories.
`fit_perception_presence.py` and `fit_perception_cup.py` write separate
components; `apply_perception_components.py` refuses failed validation and
cannot claim reserved performance. Raw HDF5, trajectories, images and
weights stay outside Git. Parent hashes are checked throughout.

Local checks:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=runtime/src .venv/bin/python -m unittest test_pour_size -v
```

Controller/geometry checks requiring Torch run CPU-only in the existing
native environment. Forty-seven local and 58 native CPU checks pass. Native checks use the staged
controller before deployment, with CUDA disabled. See `source-checks.json`.
