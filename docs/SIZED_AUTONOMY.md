# Autonomous pouring across nine cup sizes

This study combines the validated cup presence tracker and independent RGB/color
level estimates with the existing visual grasp, projected pouring motion and
side return. A checkpoint binds each of the nine declared height/diameter pairs
to the calibrated 1024px cup-detail camera and the frozen perception artifacts.

The controller acquires the cup from camera frames, tracks its lower outline
during the task and computes separate seven-frame level estimates. The higher
readout helps trigger braking. Full-cup confirmation uses six consecutive
readout pairs after settling and requires stable agreement and three
confirmations. Loss of cup evidence inhibits further tilting and brakes an
active pour. Known cup dimensions set the rim and mouth clearances. The grasp,
bottle evidence and return use RGB, joint readings and the controller's own
history. Simulator contact and liquid inventory are recorded for scoring.

The two development fixtures cover the smallest and largest cups. The nine
fresh fixtures each start with a size-matched settled prefill cut near half
height, and each targets a near-full stop. They use the same bottle and one
previously supported layout family with modest position variation. The study
tests these simulated conditions and the nine declared cup geometries.

The primary score requires a verified grasp, at least 1 nominal mL delivered,
a near-full stop at or above 90% of rim height, sampled headroom below 98%,
ordered sideways return and release, and zero measured particle loss. The
prior provisional tolerance of up to five lost particles is reported
separately. Setup fixes and failed development attempts remain in the record.

The owned launcher pins simulation to the RTX 4070 with four CPU cores of
quota, a 16 GiB memory cap, and no external network interface. It checks for
competing 4070 processes and yields its own container on contention. The
benchmark uses a separate source mount and a dedicated recordings directory.

Both [development pours](../evidence/sized-autonomy-A/development-results.json)
passed. The smallest cup finished at 96.12% of rim height and received 24.33
nominal mL. The largest finished at 93.44% and received 41.52 nominal mL.
Both had zero measured particle loss and completed the ordered side return.
The peak sampled heights were 96.45% and 93.79%. The two trials took 713 and
968 wall-clock seconds, so the guarded nine-case batch has a four-hour
deadline to accommodate completion under the same resource caps.

All nine fresh attempts completed. Eight passed the full task, and all nine
had zero measured particle loss. Every attempt transferred liquid and
completed the ordered return and release. Passing final heights were
91.69–94.83% of rim height, and the batch peak was 95.08%.

The 0.90× height and 1.00× diameter case ended with a level-confirmation
stop at 79.04% of rim height. Its final filtered RGB and color estimates were
82.49% and 78.14%, a difference of 4.35 percentage points against the
3.5-point limit. The controller kept the upright return and set-down
sequence after stopping. This case remains an underfill failure despite
zero measured loss.

The [per-size scores](../evidence/sized-autonomy-A/test-results.json) include
the stop and spill outcomes, ordered return checks, and recording hashes.
The [completion audit](../evidence/sized-autonomy-A/completion.json) confirms
all nine unique IDs used the sealed size checkpoints, the separate 4070
container exited, and the 135 original runtime sources and parent checkpoint
matched their recorded hashes. The fresh batch took 1 hour 59 minutes.

## Short-cup calibration

The short, standard-diameter cup's tracked center was within 0.4 mm of its
recorded center. Reprocessing the frames at the recorded center still showed
an RGB overestimate in both members of the RGB ensemble. The independent
color readout was closer to the measured height.

The repair fits one scalar against the measured liquid height from four
frames in the first settled interval of the failed attempt. For the 0.90×
height and 1.00× diameter cup, it subtracts up to 3.246 percentage points
from the filtered RGB level during settle. The correction ramps up between
75.5% and 79.5%, stays level through 83.5%, and ramps to zero by 89.0% of
the unadjusted RGB reading. Its mapping is monotone. The unadjusted reading
still drives braking and flow estimation. The color readout and the
3.5-point agreement limit stay independent of the fit.

On the second settled interval of the same recording, RGB mean absolute
error fell from 3.60 to 0.35 percentage points. The largest gap in the last
six frames fell from 4.38 to 1.13 points. The corrected pair permits another
top-up at that level. The correction also passed replay on all 35 archived
perception recordings for this size. The other eight size checkpoints are
byte-identical to the previous round. [Calibration receipt](../evidence/sized-autonomy-A/r1-calibration.json).

The [frozen follow-up plan](../evidence/sized-autonomy-A/r1-plan.json) calls
for a strict zero-loss short-cup verification before nine fresh fixtures.
The fresh tests use new seeds and slight position shifts while keeping
the same bottle, Material A, and approximately half-height prefill.

All [10 geometry screens](../evidence/sized-autonomy-A/r1-geometry.json)
and 30 targeted controller tests passed. The
[short-cup verification](../evidence/sized-autonomy-A/r1-development-results.json)
settled at 92.33% of rim height, peaked at 92.46% and measured zero particle
loss. The correction brought 36 of 98 affected settled frames back within
the agreement limit. At the near-full stop, the gap was 0.21 percentage
points. The bottle completed the ordered side return and release.

All [nine fresh simulated pours](../evidence/sized-autonomy-A/r1-test-results.json)
passed the strict task criteria with zero measured particle loss. The
short standard-diameter cup stopped at 95.31% of rim height and its
RGB/color gap was 0.62 percentage points. The correction brought 16
settled frames back within the agreement limit during that attempt.
Final cup heights ranged from 92.21% to 95.58%. The largest sampled
height was 95.94%, below the 98% limit. All nine grasped and completed
the ordered side return and release.

The [completion audit](../evidence/sized-autonomy-A/r1-completion.json)
checks the recorded fixture specifications, selected weights, container
limits and preservation of the original runtime. Both test containers
exited normally. The fresh batch took 123 minutes.
