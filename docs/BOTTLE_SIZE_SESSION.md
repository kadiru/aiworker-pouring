# Bottle variation: proposed eight-hour maximum session

This is a **plan, not a launched or accepted training run**. The current
18 reserved moving-perception scenes belong to the cup-size study. They run
only after the repaired RGB/color pair passes every existing development
gate. The 30 bottle-task tests below are a separate future benchmark.

The first bottle curriculum has 15 development shapes: height multipliers
0.85, 0.925, 1.0, 1.075, 1.15 crossed with body-radius multipliers 1.0, 0.95,
0.90. Start with short/nominal/tall at nominal width, then intermediate
heights, then thinner bodies. Four intermediate height/width combinations
are reserved, excluded from fitting. These are small simulated vessels:
114.75–155.25 mm high, 49.8–54.8 mm outer body diameter.

The bottom, 2.4 mm wall, 48 mm grasp height, lower grasp band and 18 mm inner
mouth diameter remain fixed. Height changes occur above 65 mm. Width never
exceeds today's bottle. A thinner bottle still needs measured contact and
closure checks; smaller does not automatically mean graspable. The proposed
authored shell mass scales at constant density from the 180 g baseline,
giving 145.3–198.1 g. This is a simulation convention, not a physical mass
calibration. Native inertia/center-of-mass validation is outstanding.

| Stage | Maximum time | Work and exit condition |
|---|---:|---|
| Readiness | 30 min | Verify data/model/source hashes, free disk, idle owned simulator, 4070 priority guard and accepted cup perception. |
| Dry geometry/grasp | 45 min | Up to 45 trials: three per shape. Check derived collision geometry, native inertia, grip/lift, full tilt, side withdrawal, placement and release. |
| Calibration | 150 min | Up to 90 training and 30 validation episodes; whole-episode splits declared before capture. Balance sizes, low/intermediate/high fills, bottle motion and layouts. Audit and review actual images. |
| Supervised fitting | 75 min | At most three predeclared candidates. Independent RGB/color and source-pose/presence models. Motion BC only if verified demonstrations are collected. |
| Development pours | 60 min | Up to 12 attempts, six full-stop and six empty-stop, only after all component/geometry gates pass. Retain failures. |
| Reserved task evaluation | 90 min | Freeze sources/weights, then 30 once-only tests, balanced full/empty. No tuning or retries. Skip if any prerequisite fails. |
| Closure | 30 min | Close recordings, preserve checkpoints and failures, audit outcomes, labeled video and GitHub commit/push. |

Total cap: **480 minutes / eight hours**, including closure. Counts are upper
bounds, not throughput promises. Measure initial trial duration and recording
size before admitting the remaining cases. Insufficient time or incomplete
data blocks dependent stages; the run must not fill its budget by repeatedly
trying the held-out tests.

Use the existing 4070 UUID, four-core quota on CPUs 16–23, low CPU shares/nice
priority and 16 GiB cap. A foreign 4070 compute job retains priority. Never
stop another job, change host drivers, touch the 4090 or automatically restart
after the guard shuts down the owned container. Shared-host overhead remains
possible despite these limits.

The future coordinator must persist a monotonic start/deadline, checkpoint
after each closed episode and every ten optimizer epochs, and keep the same
deadline after resume. Save model, optimizer, RNG, episode cursor, splits and
hashes. Admit a new trial only if its conservative maximum duration plus
recovery/closure margin fits. Reserve the final 30 minutes for closure.
The session launcher is **not implemented yet**; the machine-readable plan
specifies these requirements rather than claiming they have run.

Before spawning any resized bottle, replace fixed bottle constants together
in a separate candidate: visual/collision profile, particle initializer and
containment, capacity, mass, mouth/grasp frames, neck-to-base visual projection,
lift confirmation, IK and full-body return/placement clearance. Update
neck-relative empty-detection crops and explicit checkpoint size support.
The active runtime still rejects bottle scaling. Do not bypass that check
or simply stretch the visual mesh.

Retain Material A, the accepted cup tracker and liquid parameters. The method
remains supervised perception and contextual behavioral cloning with IK and
programmed feedback, not RL. Optimize measured liquid-height error; require
each size's MAE <2 percentage points, near-full error <3.5 points,
overestimate <2.5 points and RGB/color disagreement <3.5 points. Task goals
remain a settled cup at least 90% full or confirmed empty source, no sampled
surface above 98%, correct ordered natural return/release, and at most five
lost particles at the unchanged particle resolution. Report strict zero loss
separately. Known dimensions remain declared inputs, not inferred sizes.

Generate and verify the numerical proposal without GPU use:

```sh
python3 tools/plan_bottle_session.py
python3 -m unittest discover -s tools -p test_bottle_curriculum_geometry.py -v
```

The resulting `evidence/bottle-size-A/session-plan.json` and
`geometry-proposal.json` explicitly mark all native grasp/inertia/autonomy
acceptance as false. No recordings, grasps or successful pours are implied
by the CPU geometry calculations.

Native CPU mesh screen completed November 10: all 15 proposed development shapes pass the necessary upright open-hand/arm/table clearance and opposed-tip intersection screen at one baseline layout. First sampled opposed-tip contact is at 35–40 degrees of finger closure. This is geometric contact bracketing, not a native dynamic grasp, force/friction or liquid test. The fixed-grasp-height design requires the same upright IK target across heights. Evidence: `evidence/bottle-size-A/bottle-grasp-envelope-sat.json`.

Storage readiness: the current 36 static plus 18 possible reserved perception fixtures are estimated at about 2.5 GiB with a 50% margin. The full eight-hour capture plan needs a separate prelaunch capacity check: extrapolating the largest measured 3-second recording to 120 calibrations and 42 hypothetical 60-second development/test attempts gives about **57 GiB free**, including 3 GiB for video/checkpoints and 10 GiB retained headroom. Ubuntu had about **16.7 GiB free** at this check. This is a conservative estimate, not measured future bottle-trial size; storage readiness failed at that assessment. No files from other jobs or unique historical evidence were deleted. See `evidence/bottle-size-A/session-capacity-assessment.json`.


November 10 storage update: the verified recording migration is complete. The new ext4 partition is mounted at `/mnt/robot-training`, with AIWorker recordings under `/mnt/robot-training/aiworker-pouring/outputs`. A persistent bind mount preserves all existing host/container recording paths. All 170,480 files were verified before the old duplicate was removed; both mounts and the container read were checked. The training partition has **96.70 GiB available**, so the **56.92 GiB capacity check now passes**. This does not accept the perception models or launch training. Run `python3 /home/kadir/aiworker-pouring-4070/tools/verify_robot_training_storage.py --minimum-free-gib 57` before any future collection, then verify GPU priority and all remaining gates. See `evidence/bottle-size-A/storage-post-migration-20261110.json`.
