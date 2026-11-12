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

## First autonomous bottle-size trials

The guarded simulator trials use bottle heights of 124.875 and 145.125 mm,
with outer body diameters of 54.8 and 52.3 mm. All four combinations use
the 18 mm mouth, 48 mm grasp height, 2.4 mm wall, Material A and the same
accepted standard-size cup. Each bottle starts with 80 nominal mL and each
cup starts around half height. The trials target near-full stops.

The declared profile drives the collision and visual meshes, initial
particle grid, containment scoring, lip projection and sideways return.
The shell model authors mass, center of mass and diagonal inertia, using
the existing 180 g empty nominal bottle as its density reference. The
simulator checks the authored values and scores contact and lift on a free
rigid bottle. The actor uses RGB and measured joints for control. It
projects the initial source search region onto the new lip plane and
uses the retained visual templates, cup models and motion policy.

The first [six-fixture plan](../evidence/bottle-size-A/autonomy-r0-plan.json)
uses two nominal-width development pours followed by four fresh attempts
covering both heights and widths. All six [geometry screens](../evidence/bottle-size-A/autonomy-r0-geometry.json)
pass, including approach, lift, 48 outbound angles and six sampled starting
angles for the sideways return. First sampled opposing fingertip contact
is at 35° for nominal width and 37.5° for the narrower body.

Both [nominal-width development pours](../evidence/bottle-size-A/autonomy-r0-development-results.json)
passed on the first grasp. Both sustained bilateral contact through the
measured pour, stopped near full and completed the sideways return and
release with zero measured spill. The short and tall bottles finished
at 94.03% and 94.60% of rim height, respectively. The frozen test batch
requires both of these strict passes before it can start.

All four [fresh attempts](../evidence/bottle-size-A/autonomy-r0-test-results.json)
passed with zero measured particle loss. Each attempt verified the
grasp on the first try and recorded 19 upright bilateral-contact frames
with the bottle lifted above 30 mm before tipping. All sustained
bilateral contact throughout the sampled pour, stopped near full,
and completed sideways return and release.

| Bottle height | Outer diameter | Final rim height |
|---:|---:|---:|
| 124.875 mm | 54.8 mm | 91.87% |
| 124.875 mm | 52.3 mm | 91.86% |
| 145.125 mm | 54.8 mm | 92.89% |
| 145.125 mm | 52.3 mm | 93.91% |

The [completion audit](../evidence/bottle-size-A/autonomy-r0-completion.json)
verifies the four fixture/checkpoint pairs, authored mass properties,
free-bottle contact and lift, and resource limits. The fresh batch
took 49 minutes in the dedicated 4070 container. The scope is one
fresh simulation per geometry using declared bottle dimensions.

## Empty-stop verification

The first empty-stop batch froze the four accepted bottle checkpoints and
varied only initial inventory and seed within their four fixture layouts.
Each case started with 28 nominal mL in the source and 13.565 nominal mL
in the standard cup. All four grasped on the first try, stopped empty,
returned and released, and finished with zero source particles and zero
measured particle loss. All four had zero measured source inventory at
the stop. Stop confirmation followed the first zero-source sample by 0.15
and 0.35 simulated seconds in the short bottles and 4.85 and 5.75 seconds
in the tall bottles. This timing uses the 20 Hz recorded inventory for
scoring. The vision policy uses its retained wrist empty classifier.

The four predeclared trials ran once on the guarded 4070, after four
geometry screens passed. Both the original runtime and prior weights
matched their hashes. The separate 4090 process had the same PID and
allocation at both boundaries. [Plan](../evidence/bottle-size-A/empty-r0-plan.json),
[scores](../evidence/bottle-size-A/empty-r0-results.json),
[audit](../evidence/bottle-size-A/empty-r0-completion.json).

With the prepared source tree and archived weights at the original Ubuntu
study root, the guarded commands for a fresh working copy are:

```sh
python3 run_isolated_perception_job.py --helper prepare_bottle_empty_autonomy.py --label bottle-empty-r0-prepare --seconds 300
python3 run_isolated_perception_job.py --helper preflight_bottle_empty_autonomy.py --label bottle-empty-r0-geometry --seconds 1000 -- --study bottle-empty-autonomy
python3 run_bottle_empty_autonomy.py --study bottle-empty-autonomy --part development --label bottle-empty-r0 --seconds 6200
```

## First crossed vessel and placement pilot

After the four empty-stop passes, two 4070 development pilots combined new
cup/bottle pairs with millimetre-scale position shifts and two fill regimes.
The short, narrow bottle delivered 41.40 nominal mL into the tall, wide
cup and stopped at 93.26% of rim height. The tall, nominal-width bottle
emptied into the short, standard-diameter cup, delivering 28.00 nominal
mL and finishing at 68.17% of rim height. The empty latch followed the
first zero-source sample by 4.80 simulated seconds. Both passed first
grasp, strict zero loss and ordered return. The pilot composes previously
accepted, geometry-bound components. The two cases establish performance
on these fixtures. [Plan](../evidence/bottle-size-A/mixed-r0-plan.json),
[scores](../evidence/bottle-size-A/mixed-r0-results.json),
[audit](../evidence/bottle-size-A/mixed-r0-completion.json).

From the same Ubuntu study root, prepare a fresh `mixed-vessel-autonomy/src`
copy of the validated runtime and stage the three helpers. The preflight
and runner names below are staged copies of `tools/preflight_sized_autonomy.py`
and `tools/run_sized_autonomy.py`.

```sh
python3 run_isolated_perception_job.py --helper prepare_mixed_vessel_autonomy.py --label mixed-r0-prepare --seconds 300
python3 run_isolated_perception_job.py --helper preflight_mixed_vessel_autonomy.py --label mixed-r0-geometry --seconds 900 -- --study mixed-vessel-autonomy
python3 run_mixed_vessel_autonomy.py --study mixed-vessel-autonomy --part development --label mixed-vessel-r0 --seconds 3600
```

## Tall-bottle empty confirmation and placement coverage

In the three prior tall-bottle empty recordings, the wrist estimate often
fell below 0.97 after depletion. The controller therefore waited for its
maximum-tilt fallback. The added tall-bottle gate requires ten consecutive
20 Hz frames with an agreeing, tracked cup, measured grip and source
presence, at least 120° measured tilt, wrist empty score at least 0.92,
the existing scene model's empty score at least 0.98, and visually
estimated flow below 1 nominal mL/s. It uses camera and joint observations
at runtime. Particle inventory is used for offline measurement.

The [replay](../evidence/bottle-size-A/tall-r1-empty-replay.json) predicts
0.35–0.95 seconds after measured depletion, compared with 4.80–5.75 seconds
for the original stops, with zero source inventory at each candidate
stop. Three recorded tall-bottle near-full pours produced no new empty
confirmation. These are consumed recordings. The live gate is tested
with a separate tall, narrow bottle verification and six fresh fixture
combinations predeclared before that verification. All seven fixtures
passed the [sampled clearance check](../evidence/bottle-size-A/tall-r1-geometry.json).
Forty-five native CPU checks passed. There was no weight retraining.

The six [sealed test fixtures](../evidence/bottle-size-A/tall-r1-plan.json)
cross the four bottle geometries with six cup sizes and six table layouts.
Three target near-full and three target empty. The cup placement spans
29 mm in world X and 51 mm in Y. The tall, narrow verification must
pass the full task with zero loss and confirm empty within two seconds
of the first zero-source sample before the fresh batch can start.

The [live verification](../evidence/bottle-size-A/tall-r1-development-results.json)
confirmed empty after 1.45 seconds in the tall, narrow fixture, compared
with 5.75 seconds in the original same-geometry and same-placement recording.
The new gate had all ten votes at the latch. The bottle contained zero
particles at the stop and at the end. The grasp, sideways return, release
and strict zero-spill checks all passed.

The [fresh batch](../evidence/bottle-size-A/tall-r1-test-results.json)
completed all six IDs once. Four cases grasped, poured, reached their
assigned stop labels and completed side return. All six measured zero
particle loss, including the two localization aborts. Two passed the
full protocol:

| Test | Goal | Observed outcome |
|---|---|---|
| 00 | Empty | Tall nominal-width bottle, new gate at 0.90 s, zero source remaining |
| 01 | Near full | Cup confirmation timed out before grasp |
| 02 | Empty | Tall narrow bottle, fallback at 5.60 s, zero source remaining, 4.19-point cup disagreement |
| 03 | Near full | Short nominal-width bottle, 92.35% of rim height |
| 04 | Empty | Short narrow bottle, 0.280 nominal mL at stop and 0.146 at set-down |
| 05 | Near full | Bottle confirmation timed out before grasp |

This protocol requires zero source inventory at the empty latch and final
sample. The tall fallback in test 02 met that inventory check and failed
the 3.5-point cup agreement limit. Test 04 used the original short-bottle
path and failed exact depletion. Neither was retried. The
[closure audit](../evidence/bottle-size-A/tall-r1-completion.json) binds the
sealed plan, source, checkpoints and batch scores. The fresh run took
57.1 minutes on the isolated 4070. The original 135 runtime sources,
2,284 protected files and parent checkpoint matched their recorded hashes.

From a clean copy of the same Ubuntu study, stage the candidate source
under `tall-empty-r1/src`. Stage `preflight_sized_autonomy.py` and
`run_sized_autonomy.py` under the helper names below.

```sh
python3 run_isolated_perception_job.py --helper prepare_tall_empty_autonomy.py --label tall-r1-prepare --seconds 300
python3 run_isolated_perception_job.py --helper preflight_tall_empty_r1.py --label tall-r1-geometry --seconds 1100 -- --study tall-empty-r1
python3 run_tall_empty_r1.py --study tall-empty-r1 --part development --label tall-empty-verify-r1 --seconds 2600
python3 run_tall_empty_r1.py --study tall-empty-r1 --part test --label tall-empty-test-r1 --seconds 9000
```
