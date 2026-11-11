# Cup presence, tracking, and level agreement

The November 10 reserved run passed 12 of 18 scenes. Four cup-erasure challenges
produced false cup detections, one low-fill scene exceeded the RGB/color agreement
limit, and one weak detection latched the tracker out for 16 frames. The original
scores remain in [the prior report](../evidence/dynamic-A/reserved-gate-breakdown.json).

The November 11 repair fits a 37,337-parameter presence classifier on the RTX 4070.
Training uses native and downsampled cup crops, erased-cup proposals, and high-scoring
distractors from the same camera images. The 18 previously scored scenes enter
development as known regressions. The original 36 validation recordings are
kept out of the fit. The existing rim refiner still estimates the initial position.
After confirmation, each frame must show the cup's stationary lower outline and
either learned cup presence or a coarse proposal that agrees with the anchor.
Following a rejection, the next frame must supply both sources of current evidence
before tracking resumes.

The existing RGB and color level models each get a causal filter over seven
frames. For each branch, a separate constrained least-squares fit on the 4070
sets the weight on the current estimate relative to the recent median. Each fit
uses 1,440 frames with measured levels. At 20 Hz the window spans up to 300 ms.
The level gate checks individual error, near-full overestimation, model agreement,
and CPU/GPU parity using the filtered estimates.

The fresh scenes were declared before the presence fit. Each of the nine cup
sizes has a low-fill and near-full fixture with a new ID, cup and bottle placement,
seed, and bottle motion. They use the same calibrated vessel families and initial
particle distributions as the earlier study. The fresh capture gate requires every
development episode, original accuracy limit, missing-cup and translated-view
check, and geometry preflight to pass. Training and evaluation use the existing
isolated launcher with GPU, CPU, and memory limits.

The simulator consumes a temporary copy of the new protocol and selection while
the owned study lock is held. The collector restores the original files by hash
after the simulator stops. The recordings and model weights stay on Ubuntu. The
test scope is simulated camera perception with prefilled cups and scripted bottle
motion.

The first development replay passed 296 of 297 recordings. The classifier rejected
one frame of the high-fill training fixture SD_A_motion_train_r0_06_high. Adding
corroboration from an aligned coarse proposal and the current lower outline
restored 40 of 40 tracked frames. The full second replay passed 297 of 297
recordings and 270 of 270 translated/erased challenges across 4,833 level frames.
The prior dropout scene tracks 40 of 40 frames, and peak disagreement in the
prior disagreement scene fell from 3.9547 to 2.2216 percentage points.

The [frozen selection](../evidence/dynamic-A/followup-selection.json) binds the
candidate and fitted filters to the [development scores](../evidence/dynamic-A/followup-development.json)
and [geometry preflight](../evidence/dynamic-A/followup-geometry.json).
