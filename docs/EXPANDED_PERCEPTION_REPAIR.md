# Expanded cup localization and liquid-level repair

The November 10 request explicitly permits repairing localization as well as
liquid level. This supersedes the previous requirement to keep tracker t1
byte-identical. Old models, failure reports, and the active robot controller
remain preserved. New candidates are separate artifacts.

The development dataset contains 243 reviewed episodes: 207 training and 36
validation, split by whole episode. It includes all 36 previously collected
low/intermediate-fill scenes, including the six wide-cup localization failures.
The first two fits reuse these recordings. The 18 reserved moving
perception fixtures and the nine previously consumed static tests are excluded
from fitting. These are perception tests, not autonomous pouring attempts.

The rim model is a 52,129-parameter RGB heatmap CNN. It fine-tunes from d2 using
the first 12 full-resolution observation frames per episode, cropped around
the frozen coarse detector. It minimizes heatmap cross-entropy plus a weighted
coordinate regression loss, with AdamW and small image transformations. The
160-epoch fit has 24,960 updates. Tracker t2 uses this new fine model with the
previous coarse detector, confidence threshold, eight-frame confirmation, and
lower-cup-outline tracking rules. Full temporal replay and synthetic missing-
cup/moved-view rejection tests remain mandatory.

Once tracking passes, all level crops are regenerated from its image-derived
positions. True scene position is used only to score localization; it never
sets an inference crop. Known cup geometry remains an explicit input. Two
independent 179,409-parameter CNNs learn normalized liquid height `y = h/H`:
one sees RGB, the other a red mask plus grayscale/edge features. Both minimize
`mean((prediction-y)^2) + 0.25 * mean(largest-quarter squared errors)` using
AdamW, with balanced motion/size/episode sampling. Agreement is an acceptance
check, not a training loss. No reinforcement learning or motor-policy update
occurs in this repair.

Original acceptance limits remain: every initial/confirmed location below
3 mm error; every development episode's level MAE below 2 percentage points;
near-full maximum absolute error below 3.5 points and overestimate below 2.5;
RGB/color disagreement below 3.5 points; CPU/GPU level difference below 0.01
points. Missing color must reject. All required cases must pass, with no dropped
failures or threshold changes, before a frozen selection can authorize the 18
once-only reserved fixtures.

Fitting runs in a separate container on the RTX 4070 only, with four CPU cores
of quota, CPUs 16–23, low CPU shares/nice priority, and 16 GiB RAM. Source and
recordings are mounted read-only; only the owned study directory is writable.
The host launcher serializes jobs, checks the migrated storage, and stops only
its immutable container on competing 4070 compute use or its deadline. It does
not start the simulator. Shared-host overhead is not zero.

Native artifacts live under
`/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1`.
Raw data and model weights remain outside Git; source, hashes, and compact
result reports are versioned. Final acceptance is recorded in the accompanying
expanded-repair result report, not inferred from training completion.

The initial level candidates r8/r9 both failed the retained low-fill accuracy
checks. R8 uses image-derived XY context; r9 removes it. Each branch trained
800 epochs / 35,200 updates. The next data extension declares 36 additional
training-only static scenes at seed cuts .30/.34, between the prior .24/.38
cuts, across the same nine sizes and two existing training layouts. These are
new physical simulation recordings with measured labels, not generated image
augmentation or copied validation frames. Protocol v6 preserves the original
validation/holdout specifications and all thresholds; v5 is archived. All new
cases require physical, camera-review and tracker admission before r10 fitting.

All 36 new static recordings completed with stable measured levels of
27.9987–33.3014%, zero measured liquid loss, and accepted t2 localization.
All 144 selected native camera frames were inspected. Cache v5 contains
3,195 training images from 243 episodes and 918 validation images from 36
episodes. R10 ran 800 epochs / 40,000 updates per branch; both branches failed.
Its worst RGB episode MAE was 2.2637 percentage points. Failures are retained.

A development-only crop sensitivity audit found that exact-center cropping
reduces that low-fill RGB error from 2.2637 to 0.0884 points and color error
from 2.9888 to 0.7361 points. Other color high-fill errors persist, so localization
is a contributor rather than the sole cause. Truth crops are diagnostic only:
they never replace image-localized training, validation, or inference inputs.
The initial diagnostic hit its 16 GiB cap because the compressed array was
repeatedly loaded; loading it once fixed the problem. Both receipts are retained.

R11 keeps the same models, data splits, labels, loss and accuracy thresholds.
Instead of translating a preprocessed crop, it perturbs the image-derived XY
by a random offset within a 2 mm disk, computes the resulting camera affine
map, resamples RGB, and only then computes the independent color mask/outline.
This simulates the observed localization-error envelope and preserves the
projective change in scale/shear. Three CPU contracts verify native projection,
identity resampling, and same-world-point consistency. Two preparation/code
errors stopped before any optimizer update; their drafts and logs are archived.
Actual R11 acceptance must come from its final report, not this plan.

R11 finished with worst episode MAE 2.2746 pp RGB / 2.7560 pp color;
near-full color overestimate remained 2.7877 pp. It failed and remains archived.
The subsequent d4 rim fit also uses the new 36 calibration scenes: 243 training
and 36 validation episodes, with 12 initial images each. The 200-epoch fit
selected epoch 195. Tracker t3 passes 279/279 scenes, 5,067 sampled frames and
180/180 rejection challenges, with maximum initial/tracked error
1.6646/1.6159 mm and CPU/GPU difference 0.00212 pixels. Seven new admission
contracts reject split leakage, duplicate records, absent review and bad data.

Cache v6 rebuilds all 3,195 training / 918 validation crops from t3 predictions.
R12 fine-tunes each level branch from the lower retained development score
between R10/R11, while preserving both failures. Splits, labels, loss and gates
remain unchanged. The model initialization choice is development selection;
none of the 18 reserved images has been collected or used for this choice.

R12 remained below the required accuracy. R13 then added 6,390 correlated
training-only crops directly from the original 1024px recordings, avoiding a
second resampling of reduced crops; all 3,195 original training frames and
918 validation images were retained. Nine selected original/offset crops were
inspected. This augmentation also failed; no reserved tests were started.

R14 changes the representation: two independent convolutional models output
cup-rim and liquid-surface center heatmaps. Soft-argmax gives image landmarks;
calibrated image rays reconstruct cup XY at the declared rim height and the
liquid surface height on that vertical line. The target is still measured
normalized liquid height. True scene centers author landmark labels only and
never enter inference. Inputs remain camera RGB (or independently derived
color/outline), image-estimated crop origin, known cup geometry and calibration.

The new supervised objective is Gaussian heatmap cross-entropy plus 0.2 point
smooth-L1 plus 40 times the original MSE/worst-quartile level objective. Each
branch starts independently from random weights. It retains native training
crop alternatives and the same episode splits. Three CPU contracts verify
height round trips, crop-origin invariance and finite nonzero gradients.
The reserved evaluator differs only in model-prediction dispatch: its entire
main loop and error/scoring functions are AST-identical to the original, a
condition checked again before any selection freeze. No changed pass threshold
or autonomous motor-policy claim is implied by this new architecture.

R14 removes the earlier worst low-fill failure, but RGB remains unaccepted at
2.2179 pp worst episode MAE on a high-fill scene. Its selected color model
passes the accuracy criteria but fails CPU/GPU parity at 0.0100473 pp versus
0.01 pp allowed. Neither model pair is activated and reserved tests remain unused.

R15 adds declared height, diameter and aspect channels to feature extraction,
starting those weights at zero to preserve the original function. The fully
precise training path projected beyond its frozen time budget; its stable ZIP-
validated partial checkpoint, protocol and interruption receipt are retained.
Only that owned container was stopped. R16 repeats the same seed/initialization
with fast optimization and full-precision validation/inference. Epoch zero is
eligible for selection, preserving the initial function if fine-tuning degrades
it. Two conditioning and three precision-mode CPU contracts pass; GPU acceptance
remains determined by the final fit report. No thresholds were relaxed.


R17 tests a fixed equal-weight RGB ensemble of the R12 direct regressor and
R14 relative-landmark model, while retaining the separate R14 color-landmark
model. The averaging weights were declared before calculating the ensemble
screen; color predictions never enter RGB inference. The cached development
screen passes with worst episode MAE 1.9597 pp (RGB) and 1.8728 pp (color), and
maximum cross-check disagreement 2.2807 pp. This screen alone grants no test
permission. Actual full-precision CPU/GPU predictions, single-frame inference,
missing-color rejection, all original gates and frozen-input hashes must pass.
Packaging R17 performs zero new optimizer updates; its member CNNs were trained
on the 4070 during this repair. Full-precision inference addresses the narrow
R14 numerical-parity failure without changing its accuracy tolerance.

R16 completed both 800-epoch / 40,000-update fits. Color passes with worst
episode MAE 1.8182 pp and CPU/GPU difference 0.0000437 pp. RGB remains
unaccepted at 2.2172 pp; the pair is therefore unaccepted. Its completed
weights, reports and isolation receipt are retained independently of R17.

Actual R17 inference verification passes every original development gate:
worst episode MAE 1.9591 pp RGB / 1.8723 pp color; CPU/GPU differences
0.00003354 / 0.00004841 pp. All 918 validation frames also pass single-frame
inference checks on CPU and GPU. Missing color rejects. Frozen component
selection uses t3, R17 RGB and R17 color; the reserved evaluator changes only
prediction dispatch, with scoring/main functions AST-identical to the original.
The once-only 18-scene result is recorded below.


The reserved batch is complete: **12/18 pass overall**. Every case was recorded
and scored exactly once after the frozen development selection. All 18 native
data audits pass, with stable levels and zero measured loss. Near-full settled
labels span 94.6700–95.6428%; these are prefilled perception fixtures, not liquid
transfers. Six four-camera final pages and nine six-time-point motion pages
were actually reviewed: 72 final camera images plus 108 cup-detail motion
images. This is selected-frame review, not continuous playback.

All 720 motion frames have finite RGB/color estimates. Both branches pass their
individual accuracy limits in every scene. Aggregate RGB/color MAE is
0.5634/0.4951 percentage points; worst episode MAE is 1.2694/1.8163 points.
Near-full maximum absolute errors are 1.7220/2.4488 points and maximum
overestimates 1.1635/1.8508 points. Initial localization passes 18/18, with
maximum error 1.4824 mm. Valid tracked outputs have maximum error 1.3800 mm;
one scene loses valid tracking, so that maximum does not establish uninterrupted
tracking. CPU/GPU level discrepancy stays below 0.000051 percentage points.

The six failed scene suffixes and causes are:

| Scene | Failed checks |
| --- | --- |
| 00_high | Synthetic missing-cup rejection |
| 02_high | Missing-cup rejection and confidence margin |
| 04_high | Missing-cup rejection and confidence margin |
| 05_low | RGB/color disagreement: 3.9547 pp > 3.5 pp |
| 06_high | Synthetic missing-cup rejection |
| 07_low | Tracking rejection for 16/40 frames; confidence margin |

The retained coarse detector assigns confidence 0.4311–0.7116 in those four
synthetic erased-cup challenges, above its unchanged 0.39655 threshold. This is
a rejection failure, not evidence that the measured cup coordinates were wrong.
For 07_low, one `cup_absent_or_uncertain` rejection at 1.705 s is followed by
15 `latched_rejection` frames through 2.455 s. For 05_low at 1.805 s, true height
is 28.3996%, RGB predicts 26.3729%, and color predicts 30.3277%. These preserved
observations explain why individually passing average errors do not imply a
passing combined controller input.

`explain_reserved_perception.py` reads stored predictions only and asserts its
check-by-check explanation matches the original evaluator. Its first reporting
attempt did not handle missing tracked-coordinate values; the reporting-only
bug was fixed, with the original source and failure log retained. The evaluator
and inference were not rerun. No candidate weights, scoring thresholds, reserved
recordings or active controller were changed after test observation.

Final closure verifies 2,284 protected parent files and 135 runtime sources,
all earlier frozen fit inputs, 36 added training recordings and their review
hashes, unchanged validation/reserved specifications, and exactly 18 consumed
reserved IDs. The owned simulator is stopped (its 30-second stop grace expired,
exit 137, `OOMKilled=false`, after recording/audit completion). No AIWorker GPU
process remains. The separate 4090 process is present; no foreign process was
signaled or reconfigured. Storage has 94.96 GiB free.

The next repair targets coarse rejection with development distractor/missing-
cup examples, moving low-fill cross-check consistency and transient tracking
confidence. These 18 tests are consumed and must not be represented as fresh
again. Resized autonomous pouring and bottle-size training remain unaccepted.
