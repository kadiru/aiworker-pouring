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

A separately named static replacement collects the missing low/intermediate
fill coverage at those same two layouts. The bottle stays still, and the
original accepted moving examples remain in the dataset. Both original
and replacement geometry screens pass. Existing validation/reserved case
specifications and original accuracy thresholds remain byte-equivalent
as protocol v3 advances through the preserved v4 and v5 revisions.

The static replacement must pass native recording/label/containment checks,
actual camera review and the unchanged tracker before extending the cache.
R7 is prepared to fine-tune the independent R5 RGB/color models on that
expanded development set, retaining all original gates and the added
per-episode gate. It has not run at the time this draft was written.

Twelve CPU regression checks cover per-episode failure visibility, removing
position as an input in R6, retaining color-reference structure, absent-color
rejection and once-only evaluation metrics. A passing unit suite is not
perception or pouring acceptance. Full native datasets and weights remain
outside Git; source, small reports and hashes are versioned.

The bottle-size plan and its separate native robot-mesh screen are in
`BOTTLE_SIZE_SESSION.md`. Bottle scaling remains disabled in the active
runtime until derived geometry, dynamic contact, inertia and motion checks
are implemented and passed.
