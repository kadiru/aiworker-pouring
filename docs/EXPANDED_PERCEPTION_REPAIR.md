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
