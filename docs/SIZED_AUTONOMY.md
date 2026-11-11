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
