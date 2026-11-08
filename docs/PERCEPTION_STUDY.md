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
The twelve new geometry screens and native settled-label checks pass;
collection is complete. All twelve final native stills are inspected, with
visible low/near-full surfaces and partial bottle occlusion in held views.
There are now 73 complete records and 292 SHA-verified local scalar sidecars.
Supplemental measured levels span 7.06–97.67%.
Recorded splits stay unchanged, and the fit manifest records pilot reuse. Its targets are offline projected physical rim centers;
runtime input is RGB. The loss is focal heatmap loss plus twice smooth-L1
offset loss. Training adds integer image translations, mild photometric
changes and explicitly synthetic absent-cup masks. Old train/validation task
views are replayed alongside the new detail views. No reserved images or
autonomous outcomes have been consumed for this repair.

## Accepted baseline candidate and retained failures

After the extra data, a 600-epoch / 7,200-update localizer fit passed on
sampled views but failed its actual temporal check at 3.106 mm. A separate
240-epoch / 9,120-update refinement used all twelve early frames plus the
final image: 607/276 images, 67/28 train/validation episodes. It improved
training precision but still failed validation at 4.287 mm. Both are retained.

A fixed equal-weight average of these two 42,947-parameter models passes
both per-frame and eight-vote temporal checks. It uses the lesser confidence
and rejects model disagreement above 6 pixels. Original-cup maximum errors
are 2.576 mm training / 2.699 mm validation; no size-generalization pass is
claimed. No ensemble-weight search or reserved images were used.

The new 58,561-parameter cup CNN used 240 epochs / 4,080 updates and
1,049/368 train/validation images. Primary original-cup validation is only
four episodes / 44 images, including 33 near-full frames. Selected epoch
235 has 1.753-point MAE, 1.854-point near-full maximum error and 1.573-point
maximum overestimate. The independent linear color model failed at 6.730
points; adding image-derived view position did not improve it. Those fits
and source are preserved. Four predefined nonlinear color candidates were
then compared with CNN weights fixed: two feature sets and two native
training domains, fixed unit RBF bandwidth and .01 ridge. The selected
thirteen-color-feature model uses one prototype per 43 training episodes
and 44 coefficients/intercept; maximum validation error is 5.916 points
against the unchanged 6-point gate. It uses no level-CNN features or live
particle/object truth. Its narrow margin requires independent evaluation.

The combined candidate is
`7c872d14a588b913f9a62d30762775a0d4ee6de909571f7e4cf2fd266ad85fd1`.
Nine new original-cup holdouts span low through near-full seed cuts and
small position jitters; geometry/frustum checks pass. The original nine
size holdouts remain unstarted and deferred. Six development and six
reserved autonomous attempts remain gated behind these checks. The
candidate is deployed only to the owned simulator; no autonomous result
is claimed from component validation.

## Independent round-0 failure and round-1 repair

The nine once-only baseline checks completed with accepted stable labels and
temporal confirmations for every cup. The batch fails: maximum localization
3.147 mm (limit 3), level MAE 2.167 percentage points (limit 2), and near-full
maximum overestimate 3.300 points (limit 2.5). Near-full maximum absolute
error 3.300 points and color maximum error 5.858 points pass their respective
3.5- and 6-point gates. This is stationary perception evidence, not nine
autonomous task outcomes. See `baseline-holdout-r0.json`.

Round 0 is immutable in `rounds/baseline-r0`, including its weights, source,
plan and failed report. Each archived file is included in protected-parent
hashes. The nine records are explicitly admitted as training for round 1;
their recorded splits and old failure result do not change. Plan version 8
declares nine fresh original-cup checks before refitting, with different seed
cuts and small position offsets. Their geometry and frustum checks pass;
their images are not used in fitting. Six autonomous development and six
reserved tests remain unstarted, as do the deferred size holdouts.

A six-parameter affine correction on the frozen localizer failed at 3.159 mm
maximum training error / 2.811 mm validation; its report is retained. The
bounded fit refined detector weights using all admitted training records,
initialized from round-0 ensemble member 0. The selected single-model epoch
60 passes at 2.088 mm training / 2.075 mm validation; actual eight-vote temporal
maximum is 1.596 mm. The full fit used 240 epochs / 11,040 updates / 240.171
seconds, with 724/276 images from 76/28 train/validation episodes. Later
epochs and all resized-cup errors remain in the report; resized autonomy
remains blocked.

The round-1 level fit passes: selected epoch 118, 1.106-point MAE,
1.179-point near-full maximum absolute error, and 0.823-point maximum
overestimate. It used 240 epochs / 4,320 updates and 1,137/379 images;
the whole level preparation/fit took 42.203 seconds. The selected independent
color check is now a 27-feature ridge model (baseline native training,
ridge .01), maximum error 4.141 points. It uses explicit color features and
image-localized viewing position, with no level-CNN features or live truth.
The round-1 checkpoint is
`d229c16180e2fc06d068a898bab3dfa970913b0f6a6d9a29f7a17c9ee8a73a21`;
its local backup and tested source hashes are verified. The nine fresh
checks are now running with this candidate fixed. Current data contains 82 complete
native records; all 328 scalar sidecars are SHA-verified locally. This is not
a full local backup of raw HDF5 images or particle trajectories.

## Round-1 independent result and level-only round 2

All nine new recordings pass integrity/stability checks and confirm a cup.
Maximum localization is 0.949 mm, level MAE 1.124 points, and color maximum
error 4.529 points. The batch still fails: near-full cases 04 and 08 overestimate
by up to 2.965 / 3.718 points, exceeding the unchanged 2.5-point limit; case
08 also exceeds the 3.5-point absolute limit. All outcomes remain in
`baseline-holdout-r1.json`; its source candidate and exact artifacts are
protected in native `rounds/baseline-r1`.

A CPU diagnostic on final frames shows that exact fixture centering still
produces 2.842 / 3.669-point overestimates in those cases. Axis-aligned ±1 mm
crop offsets span only 0.550 / 0.577 points there. The main near-full bias
therefore persists without localization error. True-centered crops are
diagnostic only, never policy inputs or fitting crops.

Round 2 retains the detector weights and rejection threshold unchanged.
The nine failed records are explicitly admitted as additional level-training
examples; their original splits and scores are preserved. Nine fresh scenes
with different seed cuts and position offsets were declared before fitting,
under plan SHA
`3fe83116ac7ed49b5405d1133c18c37e01e07534430f36b61389aa994a410ac4`.
Six autonomous development and six reserved tests remain gated/unstarted.
There are 91 audited native records and 364 SHA-verified local scalar
sidecars. Both nine-scene final-frame contact sheets were inspected; that
does not constitute continuous playback or autonomous task validation.

The level-only round-2 fit passes component validation: 1.405-point MAE,
1.114-point near-full maximum absolute error and 0.681-point maximum
overestimate. All 176 admitted near-full training frames also pass the
same maximum limits, with a 2.490-point worst error. Training used
1,236/379 images, 240 epochs / 4,800 updates, selected epoch 104; total
preparation/fit time 44.674 seconds on the guarded 4070. The separate
27-feature color/view-position RBF check uses 37 baseline prototypes /
38 fitted coefficients and passes at 4.342-point maximum validation error.
The checkpoint is
`7a17cfe0f8d44826769f2ad915873be4dff3943485aadca71657e24deb1832d8`.
Its local backup is verified, runtime sources are unchanged from the
passing CPU checks, and no simulator restart was needed. All nine new
geometry/frustum checks pass; independent evaluation is now running.

Round 2's independent evaluation is now complete and passes all unchanged
gates: 9/9 confirmations, maximum localization 1.257 mm, level MAE 1.691
points, near-full maximum absolute error/overestimate 2.261 points, and
color-check maximum 4.203 points. All 100 native stationary records pass
integrity and stable-label checks; 400 scalar sidecars are SHA-verified
locally. All nine final frames were inspected. Low-fill case 00 has a bumpy
rendered surface despite its stable scalar label; this limitation is retained.

The six autonomous development attempts have started. The first stops at
`localization_unconfirmed`, with no grasp/transfer and zero measured loss.
Bottle localization confirms; cup confidence stays about .007–.011. A CPU
replay of the actual saved start frames reproduces the rejection and confirms
that detail-camera intrinsics match the model exactly. The next attempt has
grasped and begun pouring; outcomes remain pending. This is a real task
generalization check, separate from stationary perception accuracy. All six
reserved tests remain unstarted.

## Prepared learning and evaluation

The method is supervised perception plus the existing contextual motion
behavioral cloning/IK and programmed grasp/stop/return sequence. It is not
reinforcement learning or end-to-end motor learning.

- Cup localization: two supervised RGB center heatmaps and subpixel offsets,
  fixed averaging, calibrated rejection confidence and temporal consensus. Known fixture dimensions convert image
  coordinates to tabletop coordinates; size is not visually inferred.
- Level: fine-tune the cup CNN on image-localized, normalized detail-camera
  crops, minimizing weighted squared error to measured settled surface.
  An independent RBF regression over explicit color features checks agreement;
  the preceding linear and viewing-angle candidates remain rejected. Both
  current components pass train/validation gates, as quantified above. Replay
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

Completed cup data: 27 original training and 20 validation scenes over nine
declared geometries, twelve extra baseline training scenes, and four
explicitly reused calibration pilots. Only crops passing offline localization
quality checks enter level fitting; rejected views and all original records
remain retained. The first nine baseline perception holdouts failed and are
now explicitly consumed training; nine newly declared checks remain unused.
Original size holdouts remain deferred. Six development
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
native environment. Forty-eight local and 63 round-1 native CPU checks pass. Native checks use the staged
controller before deployment, with CUDA disabled. See `source-checks-components.json`.

The finite holdout observer writes host-side progress to
`/tmp/aiworker-perception-baseline-holdout-r0/progress.json` and runs the native
recording audit after all nine close. It does not queue autonomous tasks.
Run `evaluate_perception_holdout.py` inside the owned container only after
that audit completes; its once-only report is `perception-holdout.json`.
The development observer refuses to start unless that report passes for the
same checkpoint. After development, `summarize_perception_tasks.py`,
`review_perception_tasks.py` and actual manual inspection precede
`freeze_perception_candidate.py`. No final freeze or reserved run is yet claimed.

## Actual-pour development diagnosis (in progress)

Round-2 stationary perception passes, but the first two autonomous attempts
fail. Dev00 abstains at cup confidence 0.007–0.011; CPU replay reproduces it,
with matching camera calibration. Dev01 grasps, pours, stops and returns
without loss but finishes at 87.8221% height. Immediately before its full stop,
CNN output is 96.530% against an aligned observed surface of 90.010%. A
diagnostic exact-center crop gives 96.404%; the 0.879 mm localization error
does not explain the main bias. The final-sample height later falls to 87.8221%
with unchanged inventory; the separate column-height diagnostic also falls.
The cause of that physical/surface change is not established. Original scores
remain unchanged. See `development-01-level-diagnostic.json`; two selected
stop/final observation images were inspected.

Task-domain fitting helpers are prepared locally, not activated in this
batch. The proposed explicit reuse is dev00/01/04/05 for training and
dev02/03 for development validation, declared before the next fit. Failed
task outcomes retain their recorded split and scores. Passed round-2 static
holdouts stay excluded from training; fresh static and autonomous IDs will
be declared before fitting. The original six reserved tests remain unstarted.
Actual task labels use the stable preceding half second, constant inventory,
no loss or airborne fluid, and at most one point of instantaneous deviation.
They align obs[i] with samples[i-1]; future final levels never label earlier
images. The CNN fit requires separate near-full actual-pour validation,
besides the existing static/replay gates. Color prototypes separate each
recorded pause and level bin instead of averaging an entire pour. No live
particle state is added to controller inputs. All 55 local checks pass; no
new fit or task success is yet claimed.

Dev02 also finishes underfilled, at 88.5937%, with zero measured loss and a
completed return. At stopping, its CNN predicts 96.4745% against 90.8494%
observed height; exact-center diagnostic output is 96.5246% (0.902 mm
localization error). Dev03 repeats the low-confidence localization failure,
with unchanged intrinsics/extrinsics; it does not grasp or pour. These
completed cases and their failures remain part of the fixed round-2 batch.

The end-of-record code review found no deliberate fluid reset before final
scoring. The cause of the late measured-height changes remains unknown.
NVIDIA documents particle smoothing and isosurface extraction as rendering
post-processing, distinct from the dynamics; that distinction alone does
not diagnose these recordings. Reference:
[Omni Physics particle documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/particles/particles.html#post-processing-for-fluid-rendering).
No fluid or rendering settings have been changed during this evaluation.

## Closed round-2 task batch and declared task repair

All six attempts ran once with the same checkpoint and runtime. Final totals:
2/6 strict task passes, 6/6 zero measured loss, and 4/6 ordered complete
returns. Dev04 passes at 91.7749% final height; Dev05 passes with 0.180792
nominal mL remaining, below its unchanged 0.28 empty threshold. Dev00/03
retain localization failures; Dev01/02 retain 87.8221%/88.5937% underfills.
The four actual returns have no sampled stationary upright over-cup dwell
and no post-release gripper-contact samples. Their upright/transfer stages
last 3.10–3.15 seconds. Seventy-two selected native frames were inspected.

Native integrity passes all 106 records, with 100 accepted stationary labels.
All 430 scalar sidecars are SHA-verified locally. Full HDF5/trajectory backup
is not claimed. The 109.7-second, 1280×720 development reel includes all six
attempts, at labeled 2× speed with initial/final holds. Full decoding and 54
native-camera comparisons pass (maximum mean error 1.358/255); twenty
selected encoded frames and both cards were inspected. This is not
continuous manual playback. Reports and media use round-specific filenames
so later evidence cannot silently replace their provenance.

Round 2 is archived before replacing its fitting/admission helpers. Round 3
explicitly consumes the six old development records with the declared
4/2 training/validation split above. The nine passed round-2 static holdouts
are archived and remain excluded from fitting. Nine new static holdouts and
six new development realizations were declared before fitting; all fifteen
CPU geometry screens pass. The new development records save trajectories
and outside-particle diagnostics; no such simulator data enters control.
The original six reserved tests remain unstarted. The native runtime sources,
material, physics, and stop/score thresholds are unchanged.

The first round-3 locator fit is rejected: 240 epochs / 13,440 updates,
893 training and 302 validation images from 89/30 episodes; 288.916 seconds
fitting on the guarded RTX 4070. Its best baseline maximum is 3.2085 mm
(limit <3 mm), on stationary case `PV_A_validation_04_0`. Consumed task
validation cases 02/03 reach maximum errors of 1.077/1.333 mm, but that
improvement does not waive the stationary failure. All-size validation
maximum is 81.5315 mm; resized autonomy remains gated. Failed weights,
reports, data manifest, plan and fitting/admission sources are preserved in
`rounds/task-r3-locator-attempt1`; active `best.pt` remains round 2.

Plan version 11 adds eight training-only fixtures before the next fit: six
intermediate fills at jittered existing training layouts, plus two low-fill
initial views derived from the consumed training layouts. All validation,
fresh holdout/development, and reserved identities are unchanged. The six
intermediate cases use subsets of the same settled seed; the two initial
views use the autonomous initial grid. Both require native stable-label and
containment checks. No geometry, fluid, rendering or acceptance limits are
relaxed. Declaration: `task-supplement-preparation.json`; plan SHA
`d6c98cb1bcac547ce6a611237bbb6c4b358bf646aea93bf558cd2302d28c8ed2`.

The eight additional recordings pass native integrity, zero-loss containment,
and stable labels. Final measured heights span 7.06%–86.04%; seed cuts are
initialization choices, not imposed labels. Eight final three-camera native
composites were inspected, not continuous playback. The study now has 114
complete recordings, 108 accepted stationary labels and six autonomous
attempts; all 462 scalar sidecars are SHA-verified locally. Original raw
HDF5/trajectory files remain on Ubuntu. The second 240-epoch locator fit
starts from the same preserved round-2 weights, using the augmented training
set and unchanged validation cases/gates.

The supplemented locator passes baseline component gates at selected epoch
100: validation/training maxima 2.3901/2.7317 mm. The bounded run used
997/302 images from 97/30 episodes, 240 epochs / 15,120 updates and 313.453
seconds fitting. Some later checkpoints regress sharply; only eligible
weights are retained. All-size validation maximum is 81.8389 mm, and
resized autonomy remains gated. Fresh-scene evaluation is still required.

The first actual-pour level fit fails and is preserved in
`rounds/task-r3-level-attempt1`. The offline frame selector incorrectly
admitted stable post-stop return images, including fully occluded liquid.
The independent color readout returned zero in those frames; selected native
dev02 images 650/686 confirm visible pre-stop liquid versus bottle occlusion
during return. No recorded task outcome changes. Plan version 12 declares
pre-stop settling supervision only, keeping the same episode splits and
acceptance limits. This retains 48/39 training frames from dev01/04 and
45 validation frames from dev02 (21 near full). Empty-stop dev05 supplies
no such level frames. All 56 local checks pass, including explicit rejection
of post-stop images despite stable particle labels. The corrected level
fit is a development repair, not an independent evaluation.

The pre-stop raw-CNN refit also fails its near-full gates. Its independent
color check now passes at 4.193 points maximum; the occluded-frame admission
bug is fixed, but model generalization remains inadequate. This second failed
fit and all its original sources are preserved in `rounds/task-r3-level-attempt2`.

Plan version 13 declares a learned final stop-level readout on the frozen
second-attempt CNN: two predefined feature sets with four ridge penalties,
and one 32/16 tanh residual network trained for 1,200 updates. Inputs are
RGB-derived surface features, the CNN estimate and image-localized cup XY.
Episode-balanced weighted squared error trains on the same admitted data;
validation selects only candidates satisfying the unchanged level limits,
including separate actual-pour and training near-full checks. No hand-set
output bias or new runtime source is introduced. The existing runtime
`pour_settled_model.predict` decodes the learned readout. Independent static
evaluation now explicitly scores that same final estimate and retains raw
CNN error separately. CPU/GPU and float32/float64 prediction agreement are
checked before accepting the component. All original fresh holdout, development
and reserved specifications remain unchanged and unstarted.

The first readout run reaches eligible validation metrics but stops at the
CPU/GPU precision assertion before emitting final weights. Its source,
input manifests and failure reason are archived in
`rounds/task-r3-readout-attempt1`; the original fit log is retained locally.
The same seeded, bounded algorithm is rerun with diagnostics saved even
when the numerical check fails. No candidate has been activated.

The diagnostic confirms a 0.0720-point CPU/GPU difference, caused by
floating cumulative quantiles choosing adjacent rows at an exact boundary.
A continuous-feature ablation resolves numerical parity but misses the
2.5-point overestimate limit at 2.600 points; it remains rejected. Plan
version 15 instead recovers integer counts from the 38-pixel binary-mask
histogram and computes quantiles with exact integer comparisons. Feature
count remains 709; legacy identity readouts are unchanged. Sixteen native
regressions pass, including exact-boundary CPU/GPU agreement, natural
return and legacy-readout compatibility. An initial missing test-module
error is preserved separately and corrected before the complete run.

The resulting readout passes: 5,026 fitted parameters, 139 inputs, fixed
32/16 tanh layers, selected update 980 of 1,200; 1,411/424 training/validation
images from 94/26 episodes, including 87/45 pre-stop actual-pour images.
Validation MAE is 1.392 points, near-full maximum error 2.592 points, maximum
overestimate 1.880 points, and actual-pour MAE 1.062 points. The independent
color maximum is 4.193 points. Maximum CPU/GPU prediction difference is
0.00296 points; float32/float64 difference is below 0.000005 points. Total
preparation/fitting takes 30.036 seconds on the guarded 4070. Raw CNN errors
remain reported; the final runtime readout is the accepted estimator.
Fresh-scene and autonomous evaluations are still required.

Accepted components combine to checkpoint
`595aeb7e478b165aa3c50aea4eba35358f3b7486e579838f0684417d9abdb7a7`,
SHA-verified locally (2,876,273 bytes). Only `pour_settled_model.py` changes
in the active runtime. An identity-checked idle restart deploys it to the
same owned container, retaining the 4070 binding, four-core quota, CPU set,
16 GiB cap and existing priority guard. No foreign process is signaled or
reconfigured. All deployed source hashes match the candidate. The nine
predeclared fresh perception cases start in run `20261108T232827Z`; no new
autonomous success is yet claimed.

The nine fresh round-3 static scenes all pass. Maximum cup localization is
1.2585 mm; level MAE 1.182 points; near-full maximum absolute error 1.610
points and maximum overestimate 0.730 points. Independent color maximum
error is 3.273 points. Measured labels span 7.06%–95.09% rim height; 84
near-full frames are scored. All nine final native three-camera composites
were inspected. The audit now covers 123 recordings (117 stationary plus
six prior autonomous), with 498 scalar sidecars SHA-verified locally.
Raw HDF5 and trajectories remain on Ubuntu.

The six predeclared `PV_A_development_r1_00..05` attempts have started in
run `20261108T232827Z`, observed by `/tmp/aiworker-perception-development-r3`.
The model, sources and criteria remain fixed throughout this batch. Only
six strict passes, six ordered returns and actual camera review permit
the original six reserved tests; cup/bottle size variation remains gated.
