# Low-fill and color repair, November 10

This follow-up preserves tracker t1 exactly. Its checkpoint SHA is
`f378b47db56a39dd085a7a78874c5b75e1819fdf666ca1357eb54d1a5239dcbe`.
The active robot controller is unchanged. The 18 reserved moving-perception
fixtures remain conditional on a fully passing development pair; they are
not 18 autonomous pouring tasks.

The initial crop cache contains 171 training and 36 validation episodes:
2,403 and 918 image crops respectively. Every crop uses the accepted visual
tracker and declared cup dimensions. A counterfactual diagnostic on eight
native frames showed that exact-pose crops reduce several shared low-fill
errors. Exact-pose images are diagnostic only and are excluded from the
training cache. The original cache and its source hashes are preserved.

R5 replaces the narrow masked readout with two independent 179,409-parameter
CNNs. RGB uses the full 96×96 crop. The color branch receives an integer red
mask, grayscale vessel outline and edge magnitude; it rejects an absent
red signal. It is an independent model from the same camera, not an
independent sensor. Neither model receives the other's weights or outputs.

For measured normalized height `y = h/H`, both learn `fθ(image, geometry,
image-derived position)` by minimizing
`mean((fθ-y)²) + 0.25 × mean(worst-quartile squared errors)` with AdamW.
Sampling balances static/moving groups, cup size and episode. Crop shifts
of ±2.5 pixels improve robustness to image offsets. This is supervised
regression, not reinforcement learning or end-to-end motion learning.

R5 trained each branch for 800 epochs / 30,400 updates. RGB selected epoch
450 and passes the original aggregate gates: worst-size validation MAE
1.8661 percentage points. Color selected epoch 280 and fails at 2.0202
points against the unchanged 2-point limit. Its maximum near-full error
is 2.1572 points and maximum RGB/color disagreement is 3.0535 points.
CPU/GPU parity and missing-color rejection pass. Total wall time was
400.88 seconds. Individual low-fill episode errors remain too large.

R6 tests removing absolute table position and increases crop shifts to
±3.5 pixels. It also requires every development episode's mean error to
be below 2 points, preventing accurate high-fill samples from hiding a
bad low-fill episode. Both independent branches trained for 1,000 epochs /
38,000 updates and failed. Selected RGB/color worst-episode errors are
4.0251 / 4.2210 points; maximum disagreement is 6.8782 points. Total wall
time was 540.16 seconds. Failed weights, results and complete frozen input
copies remain archived. No reserved observations entered either fit.

The data inspection also found four cup sizes whose nominal “low” moving
training case actually has a medium measured fill. File names are never
used as labels. An additive 36-scene protocol attempted two new layouts
and low/intermediate seed cuts across all nine cup sizes. The first two
closed cases triggered the frozen coarse detector's rejection during
bottle occlusion. Six actual recorded frames were reviewed. The batch was
stopped, preserving four complete recordings, one aborted recording and
31 unstarted IDs. None are admitted to fitting and no IDs are retried.

A separately named static replacement collected the missing low/intermediate
fill coverage at those same two layouts. The bottle stays still, and the
original accepted moving examples remain in the dataset. Both original
and replacement geometry screens pass. Existing validation/reserved case
specifications and original accuracy thresholds remain byte-equivalent
as protocol v3 advances through the preserved v4 and v5 revisions.

All 36 static recordings completed once and pass recording, stable-label and
containment audits, with zero measured loss. Measured levels span
20.6323–37.2887%. All 144 selected final camera frames were inspected on
12 pages; this is not continuous playback or autonomous pouring.

The unchanged tracker passes only **30/36** new scenes. All six failures
have the widest declared cup diameter (1.1×); five stable anchors exceed
3 mm and a sixth has a passing stable anchor but failing raw initial
detections. Worst stable/raw errors are **4.7044 / 4.8195 mm**. All 396
sampled tail frames report temporal validity: stable tracking does not
establish accurate localization. These cases expose an extension beyond
the original 207-episode tracker acceptance, not a changed checkpoint.

The all-case admission assertion stopped the pipeline before writing cache
v3 or preparing/fitting R7. R7 helpers exist, but **R7 has not trained**.
No failing frames or cases were discarded, no thresholds were weakened,
and all 18 reserved cases remain unused. No model or runtime source was
activated. The requested low-fill/color repair is **not accepted**.

Next dependency: resolve this low-fill/wide-cup localization limitation
before expanding the admitted training cache. Keeping the current tracker
byte-identical cannot make these six recorded anchors pass. A separate
tracker repair would change the user's current constraint; restricting new
calibration to previously supported layouts would narrow the claim and
still would not validate these failed scenes. Preserve both alternatives
explicitly rather than silently selecting away failures.

The closure audit rechecks the 36 native recording hashes, all review pages,
every retained complete/partial moving record, R5/R6 frozen inputs and
weights, 2,284 protected parent files, 135 runtime sources and unused test
IDs. Its executable is
`/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1/close_level_repair.py`
on Ubuntu. It writes an exclusive receipt and refuses to overwrite it.
Inspect the saved result with
`ssh kfuUbuntu 'cat /home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1/level-repair-closure.json'`.

Twelve CPU regression checks cover per-episode failure visibility, removing
position as an input in R6, retaining color-reference structure, absent-color
rejection and once-only evaluation metrics. A passing unit suite is not
perception or pouring acceptance. Full native datasets and weights remain
outside Git; source, small reports and hashes are versioned.

The bottle-size plan and its separate native robot-mesh screen are in
`BOTTLE_SIZE_SESSION.md`. Bottle scaling remains disabled in the active
runtime until derived geometry, dynamic contact, inertia and motion checks
are implemented and passed.
