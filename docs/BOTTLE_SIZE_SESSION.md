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

## Mixed-layout localization and stop repairs

The earlier six mixed trials supplied four development failures. A single
bad cup frame had permanently rejected an otherwise consistent rim track.
The new controller starts a fresh confirmation window after an initial
failure, within the two-second localization deadline. Confirmation
requires eight agreeing detections. The archived rim recovered at frame
10 with 0.25 mm error, and the 40 synthetic erased-cup frames stayed
unconfirmed.

Six recorded tall-bottle necks supplied a bank of signed, clipped
gradient templates. The locator starts with the original position
estimate. The new bank corroborates bottle presence and supplies a neck
position when the original score falls below its threshold. Across three
archived episodes excluded from template fitting, including the failed
grasp fixture, consensus error stayed below 0.17 mm. All 108 synthetic
erased-bottle frames were below the new presence floor. Each live grasp
still requires the established multiframe localization and hold checks.

Bounded least squares fitted one 3.12-point RGB height correction from
31 recorded frames for the 1.10×-height, 0.90×-diameter cup. It applies
only during the coarse pour at 115° commanded tilt or higher and tapers
to zero outside the midfill band. The controller uses the unadjusted
estimate for braking and a 3.5-point RGB/color agreement gate. On the next 81
frames from the same episode, the maximum RGB/color gap fell from 4.65
to 1.53 points. The stop-time gap fell from 4.19 to 1.08 points.

The short bottle's wrist score had voted empty with 48 particles still
inside. It now uses the existing maximum-tilt drain confirmation with
24 consecutive hold, visibility, low-flow and empty-evidence frames
after reaching the cap. Runtime decisions use RGB and measured
joints. Independent simulated fill, contact and particle inventory
supply training labels and evaluation measurements.

Copy `tall-empty-r1/src` into `layout-repair-r2/src` under
`/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1`.
Overlay the revised actor, vision, sized-perception and controller test
sources there. Stage `prepare_layout_repair.py` and copy the standard
preflight and runner under the helper names below. The fit reads the
archived native HDF5 recordings and earlier sealed checkpoints from
the same study. This first plan required all four development fixtures
to pass strict scoring before it would admit the six fresh layouts.

```sh
python3 run_isolated_perception_job.py --helper prepare_layout_repair.py --label layout-r2-prepare-final --seconds 1200
python3 run_isolated_perception_job.py --helper preflight_layout_repair_r2.py --label layout-r2-geometry --seconds 2200 -- --study layout-repair-r2
python3 run_layout_repair_r2.py --study layout-repair-r2 --part development --label layout-repair-dev-r2 --seconds 6500
python3 run_layout_repair_r2.py --study layout-repair-r2 --part test --label layout-repair-test-r2 --seconds 11500
```

The short, narrow development bottle emptied completely. Two final
droplets crossed the far rim while the bottle was still tipping at
about 127°, 2.8 seconds before the return. The next candidate sets the
mouth back by 8 mm in the pour direction, with a smooth ramp from 123°
to 126°. This applies only to that short, narrow bottle checkpoint.
The runner requires the short-bottle verification to pass the original
grasp, exact depletion, perception, and return checks. It admits up to
five lost particles for this development gate, and reports strict
zero-loss performance separately. The six reserved fixture IDs and
positions carry forward.

Copy `layout-repair-r2/src` to `layout-repair-r3/src` and overlay the
final actor, motion, and controller test sources. Stage the final
preflight and runner under the helper names below.

```sh
python3 run_isolated_perception_job.py --helper prepare_layout_tail.py --label layout-r3-prepare --seconds 600
python3 run_isolated_perception_job.py --helper preflight_layout_repair_r3.py --label layout-r3-geometry --seconds 1800 -- --study layout-repair-r3
python3 run_layout_repair_r3.py --study layout-repair-r3 --part development --label layout-repair-dev-r3 --seconds 2200
python3 run_layout_repair_r3.py --study layout-repair-r3 --part test --label layout-repair-test-r3 --seconds 11500
```

The [first live development batch](../evidence/bottle-size-A/layout-r2-development-results.json)
passed 3/4 under strict scoring and 4/4 within the five-particle
tolerance. The [short-bottle verification](../evidence/bottle-size-A/layout-r3-development-results.json)
then passed with zero measured spill and zero source particles at the
stop and set-down. Seven final geometry screens and 50 focused CPU
controller checks passed.

All six [fresh autonomous trials](../evidence/bottle-size-A/layout-r3-test-results.json)
passed strict scoring with zero measured loss. The tests used four
bottle geometries, six cup sizes and six new placements in the same
layout family. The cup positions spanned 23 mm in world X and 46 mm in
world Y. Each test grasped on the first attempt and returned the
bottle to the table. Near-full tests ended at 92.16–94.87% of rim
height. Empty tests had zero source inventory at the stop and set-down.

| Fresh fixture | Stop | Final rim height | Depletion to stop |
|---|---|---:|---:|
| 00, tall nominal-width bottle | Empty | 68.12% | 1.85 s |
| 01, tall narrow bottle | Near full | 94.87% | |
| 02, tall narrow bottle | Empty | 69.89% | 5.85 s |
| 03, short nominal-width bottle | Near full | 94.05% | |
| 04, short narrow bottle | Empty | 58.91% | 2.80 s |
| 05, tall nominal-width bottle | Near full | 92.16% | |

The tall, narrow-cup empty fixture had a calibrated stop-time
RGB/color gap of 0.80 points. The raw gap on that frame was 3.92
points. Fixture 05 used the neck fallback at localization. The
[closure audit](../evidence/bottle-size-A/layout-r3-completion.json)
verifies the native recordings, frozen sources and checkpoints, and
the 4070-only container limits. The fresh batch took 93.6 minutes.

## Tall-bottle timing and wider swapped layouts

The late tall-bottle run waited for the raw scene score to reach 0.98.
The revised gate can also accumulate ten consecutive votes when the
cup-aligned scene score reaches 0.84, the raw score reaches 0.65, and
the wrist score reaches 0.90. Each vote also requires encoder hold,
confirmed source presence, a tracked cup with agreeing level estimates,
at least 120° measured tilt, and estimated flow below 1 nominal mL/s.
The raw-scene confirmation path uses its original 0.98 scene and 0.92
wrist thresholds.

The archived replay covers nine tall-bottle empty pours and seven
near-full pours. It predicts 0.95 s for the earlier 5.85 s stop, with
zero source inventory at each predicted empty stop and zero false
empty stops on the near-full cases. In a live repeat with a new seed,
the gate confirmed empty after 0.95 s with zero particles in the
bottle at the stop and set-down. Two droplets escaped during the pour,
0.60 s before the stop. The development criterion admits five lost
particles. Strict evaluation scores zero measured loss.

The wider matrix pairs two exactly reversed vessel positions per
layout. It crosses five table-layout families, nine known cup sizes,
four known bottle geometries, and fifteen trials for each stop goal.
The positions start from the prior wide-layout families with
independently shifted endpoints and fresh episode seeds. Each pair
shares vessel sizes and reverses the physical positions. The paired
scenes target opposite stop conditions. The two orientations are scored
separately in the results. Motion and perception use camera observations,
joint measurements, stored calibration and declared vessel geometry.
The side return rises earlier during the same sideways uprighting motion.
This cleared a finger-to-rim overlap found during geometry screening.
One far-edge pair still had an overlap with its original wider cup.
It uses the smaller 0.90× height and 0.90× diameter cup and the short
nominal-width bottle at both original endpoints. All nine cup sizes and
all four bottle geometries remain represented. Both affected scenes are
screened again. The other 32 screen results are reused after checking
identical specs, checkpoints and runtime sources.

The first wide development batch found two additional faults. A far-side
near-full pour used a cup center 8.55 mm from the actual rim center and
stopped at 73.19% of rim height. Fitting the visible rim, base and fixed
Material A band near the neural proposal reduced that error to 0.56 mm.
Across 28 archived recordings spanning all nine cup sizes, the largest
corrected anchor error was 1.17 mm, and no synthetic missing-cup
challenge confirmed a cup. At runtime, the camera image and known
geometry supply the correction.

In the other failure, an early bottle occlusion made the cup appear
nearly full at 42° tilt. The revised controller starts using level
feedback at 80°. The earliest observed physical outflow in 23 archived
pours was 87.11°. Side return can stretch the same path when a measured
starting pose would exceed the 0.095-rad step limit. The archive and
both original failures are retained.

The far swapped repeat then emptied physically while both scene
classifiers stayed below threshold. The wide controller adds a separate
confirmation path with fourteen consecutive frames at or above 0.94
wrist probability. It also requires a measured grip, source presence,
120° measured tilt, agreeing cup estimates and less than 1 nominal
mL/s of visual flow. In archived replay, the two swapped stops would
confirm after 0.70 s and 1.20 s. Every replayed latch occurs at zero
source particles. The late spill in the farther repeat began 0.20 s
after depletion, before the proposed empty confirmation.

Run from the native study root
`/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1`.
Stage the timing runtime under `tall-aligned-r0/src`, based on the
previous `layout-repair-r3/src`. Copy that runtime to
`wide-mixed-r1/src` and apply the wide localization, return and test sources.
Use separate copies of the preflight and runner as shown below.

```sh
python3 run_isolated_perception_job.py --helper prepare_tall_aligned_autonomy.py --label tall-aligned-prepare-r0 --seconds 500
python3 run_isolated_perception_job.py --helper preflight_tall_aligned_r0.py --label tall-aligned-geometry-r0 --seconds 450 -- --study tall-aligned-r0
python3 run_tall_aligned_r0.py --study tall-aligned-r0 --part development --label tall-aligned-dev-r0 --seconds 2000
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_autonomy.py --label wide-mixed-prepare-r1 --seconds 700
python3 run_isolated_perception_job.py --helper preflight_wide_mixed_r1.py --label wide-mixed-geometry-r1 --seconds 3900 -- --study wide-mixed-r1
cp -a wide-mixed-r1/src wide-mixed-r2/src
python3 run_isolated_perception_job.py --helper revise_wide_mixed_clearance.py --label wide-mixed-prepare-r2 --seconds 700
python3 run_isolated_perception_job.py --helper preflight_wide_mixed_r2.py --label wide-mixed-geometry-r2 --seconds 500 -- --study wide-mixed-r2 --reuse-study wide-mixed-r1
python3 run_wide_mixed_r2.py --study wide-mixed-r2 --part development --label wide-mixed-dev-r2 --seconds 6500
# Stage outline and sized-perception changes under wide-mixed-r3/src.
python3 run_isolated_perception_job.py --helper check_wide_cup_outline.py --label wide-cup-outline-replay --seconds 1100
# Copy the source and replay into wide-mixed-r4 and overlay the final actor and return.
python3 run_isolated_perception_job.py --helper prepare_wide_outline_autonomy.py --label wide-mixed-prepare-r4 --seconds 700
python3 run_isolated_perception_job.py --helper preflight_wide_mixed_r4.py --label wide-mixed-geometry-r4 --seconds 3900 -- --study wide-mixed-r4
python3 run_wide_mixed_r4.py --study wide-mixed-r4 --part development --label wide-mixed-dev-r4 --seconds 6500
# Copy wide-mixed-r4/src to wide-mixed-r5/src and apply the final stop-gate sources.
python3 run_isolated_perception_job.py --helper prepare_wide_wrist_cup_autonomy.py --label wide-mixed-prepare-r5 --seconds 700
python3 run_isolated_perception_job.py --helper preflight_wide_mixed_r5.py --label wide-mixed-geometry-r5 --seconds 500 -- --study wide-mixed-r5 --reuse-study wide-mixed-r4
python3 run_wide_mixed_r5.py --study wide-mixed-r5 --part development --label wide-mixed-dev-r5 --seconds 2200
for shard in 1 2 3; do
    python3 run_wide_mixed_r5.py --study wide-mixed-r5 --part test --shard "$shard" --label "wide-mixed-test-r5-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py
```

The [farthest swapped live repeat](../evidence/bottle-size-A/wide-mixed-r5-development-results.json)
confirmed empty in 1.20 s with zero source particles at stop and
set-down, zero measured spill and bilateral grasp throughout the pour.
The final candidate passed 53 focused controller checks and 31 geometry
screens, including the 30 reserved layouts.

The [final 30-trial audit](../evidence/bottle-size-A/wide-mixed-r5-completion.json)
retains every outcome and the native recording hashes. All three
serial batches exited normally after 5.53 hours total.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Zero measured spill |
|---|---:|---:|---:|---:|
| Normal | 15 | 10 | 12 | 15 |
| Swapped | 15 | 8 | 11 | 13 |
| Near-full target | 15 | 12 | 13 | 15 |
| Empty target | 15 | 6 | 10 | 13 |
| All trials | 30 | 18 | 23 | 28 |

The zero-spill column includes seven pre-grasp rejections. Across the
23 actual pours, 21 measured zero loss and two lost two particles each.
The seven rejected trials all failed bottle-neck consensus. Among seven
fresh tall-bottle empty targets, five confirmed empty, with lags of
0.40, 0.45, 0.55, 0.60 and 5.80 seconds. One rejected bottle localization,
and one physically emptied and returned after a progress timeout.
The 5.80-second case stopped on the drain dwell and failed strict
scoring because the two cup estimates disagreed at its stop.

## Bottle localization and exact-empty retraining

The previous wide benchmark supplied the seven pre-grasp failures and
the two tall-bottle cases that took 5.80 s or timed out. The new neck
detector uses a supervised task-camera heatmap to propose a pixel and
a signed-gradient template bank to refine it. A proposal must score
at least 0.65, its template match at least 0.68, and the two locations
must agree within six pixels. The legacy locator covers frames that
fall below these checks. Grasp initialization still uses temporal
consensus. The fusion passed all 48 archived recordings, including
the seven fitting cases that previously timed out. Their position
errors ranged from 0.05 to 0.19 mm. The largest error across the
archive was 1.17 mm. All 288 synthetic erased-bottle frames rejected
the learned path. The initial standalone heatmap and the refined
heatmap without the legacy locator both missed their acceptance gates.

The wrist CNN was fine-tuned on 1,269 wrist crops from 16 episodes.
Binary cross-entropy used an exact zero-particle label and balanced
the weights by episode and class. The selected epoch was 35. The
trained path requires eight consecutive 20 Hz frames at 0.82 or
higher, a verified held source, at least 120° of measured tilt,
tracked cup history, estimated flow below 1.25 nominal mL/s and a
cup-level upper estimate below 85%. The existing camera-confirmation
paths remain available. The archived replay found 0.30 s for both
target cases and zero premature latches across 18 episodes. Ground
truth labels the training examples and scores the replay. The
controller reads RGB, encoder signals and fixed camera calibration.

A separate 32-frame calibration addresses the short standard-diameter
cup in its late coarse-pour phase. Its later 85 recorded frames show
maximum RGB/color disagreement falling from 4.86 to 2.18 percentage
points. Two vessel combinations use an 8 mm late mouth setback after
the preceding tests recorded rim-crossing droplets.

The fresh fixtures use the previous 15 wide position pairs, with
independent 1.2–2.8 mm offsets per endpoint coordinate and new episode
seeds. They retain the exact normal/swapped pairing, nine cup sizes and
four bottle geometries. The targeted development cases cover the two
slow or missed stops, a prior localization failure, and a prior spill.
The test release requires all four development cases within the
five-particle tolerance and the two targeted empty stops within 2 s.
Every fresh trial is scored against zero measured loss.

From the native study root, copy `wide-mixed-r5/src` to
`wide-mixed-r6/src` and overlay the revised actor, vision, size
perception, source watchdog and test files. Use the existing 4070-only
launcher and stage the checked-in helpers under the names below.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r6.py --label wide-r6-fit --seconds 5300
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r6.py --label wide-r6-prepare --seconds 700
python3 run_isolated_perception_job.py --helper preflight_wide_mixed_r6.py --label wide-r6-geometry --seconds 4000 -- --study wide-mixed-r6
python3 run_wide_mixed_r6.py --study wide-mixed-r6 --part development --label wide-r6-dev --seconds 6500
for shard in 1 2 3; do
    python3 run_wide_mixed_r6.py --study wide-mixed-r6 --part test --shard "$shard" --label "wide-r6-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r6
```

The [development results](../evidence/bottle-size-A/wide-mixed-r6-development-results.json)
pass all four cases with zero measured loss. The two targeted empty
stops confirm 0.35 s and 0.70 s after zero source inventory. The
former localization failure grasps with the heatmap and finishes at
92.73% of rim height. The prior spill fixture empties cleanly with
the mouth setback. All 34 geometry and 54 controller checks pass.
Two scans agreed on their first 30 checks before a near-tangent
linear program returned an indeterminate numerical status. The
resumed screen verified the remaining four cases by translating and
scaling the constraints and minimizing their common slack.

The [fresh benchmark and checksum audit](../evidence/bottle-size-A/wide-mixed-r6-completion.json)
cover all 30 autonomous trials, scored once using the frozen candidate.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Zero measured spill |
|---|---:|---:|---:|---:|
| Normal | 15 | 11 | 14 | 13 |
| Swapped | 15 | 12 | 14 | 15 |
| Near-full target | 15 | 13 | 14 | 15 |
| Empty target | 15 | 10 | 14 | 13 |
| All trials | 30 | 23 | 28 | 28 |

The zero-spill column includes two pre-grasp rejections. The 28
verified grasps completed the ordered return, and 26 of those pours
measured zero spill. Two coarse pours lost two particles each before
the empty latch. All seven fresh tall-bottle empty targets reached zero
source inventory and confirmed in 0.25–0.70 s, with median 0.35 s.
Two of those seven incurred the measured spill and another failed the
cup-agreement scoring check. The remaining failures were two bottle
localization rejections and two level-unconfirmed stops, one underfilled
near-full target and one short bottle with 0.338 nominal mL remaining.

## Wide-layout bottle, level and spill repair

The preceding 30-trial run exposed two short-bottle detections whose signed
gradient scores were too weak for eight-vote localization. One legacy match
also placed the neck more than 110 pixels away. The RTX 4070 fine-tuned the
neck heatmap with 297 labeled and synthetic-erasure images from 27 consumed
episodes. Seven other episodes were held out of this fine-tune. The runtime
still requires a proposal score of 0.70, a signed-gradient match of 0.71, at
most six pixels between them and the existing temporal consensus. The new
bank has 140 templates. On 34 native and 18 translated replays, the maximum
consensus error was 1.51 mm. The two repair cases had 20/20 heatmap matches
each and native errors of 0.10 and 0.24 mm. The learned path accepted none of
the 680 synthetic erased frames. See the
[localization replay](../evidence/bottle-size-A/wide-mixed-r7-neck-fusion.json).

The RGB height calibration minimizes weighted squared error against
timestamp-matched particle-surface height. It fits a bounded amplitude for a
piecewise-linear correction by cup size and phase. The three fits use 545
training frames and 577 replay frames from consumed recordings, with the final
six frames held out of each fit. The stop gaps fell from 4.69 to 1.34, 4.87
to 1.52 and 5.45 to 2.89 percentage points on the three repair cases. The
color branch, cup tracker, 3.5-point agreement requirement and near-rim
uncorrected guard are unchanged. The exact-empty wrist checkpoint is carried
forward from the previous run. See the
[level replay](../evidence/bottle-size-A/wide-mixed-r7-level-fit.json).

The two tall-bottle spills began with far-side rim rebounds during coarse
pouring. Their affected bottle/cup geometries use a smooth 9 mm mouth setback
and 4 mm lowering from 115° to 119°. The controller and the geometry screen
share the same pose settings. The [sealed plan](../evidence/bottle-size-A/wide-mixed-r7-plan.json)
declares seven targeted development repeats and 30 new fixtures, with
independently shifted endpoints and fresh simulation seeds. The test fixtures
span nine cup sizes and four bottle sizes in 15 exact normal/swapped pairs.

From the native study root, copy `wide-mixed-r6/src` to `wide-mixed-r7/src`
and overlay the changed runtime source and test files. Stage
`fit_wide_mixed_r7.py`, `prepare_wide_mixed_r7.py`,
`preflight_sized_autonomy.py`, `run_sized_autonomy.py`, and
`close_wide_mixed_autonomy.py` in the study root. The fit also imports
the existing `fit_wide_mixed_r6.py`.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r7.py --label wide-r7-fit --seconds 4200
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r7.py --label wide-r7-prepare --seconds 300
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r7-geometry --seconds 4600 -- --study wide-mixed-r7
python3 run_sized_autonomy.py --study wide-mixed-r7 --part development --label wide-r7-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r7 --part test --shard "$shard" --label "wide-r7-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r7
```

The first targeted repeat localized and grasped the short bottle, then
stopped low because the cup outline selected the decorative band and shifted
the rim center by 9.53 mm. For the 1.00× height / 0.90× diameter cup, the
revised tracker rejects an outline displacement above ten pixels when the
coarse and fine neural estimates agree within six pixels and still pass
presence verification. Across five full-resolution replays, the three
control centers were unchanged, both failing centers improved to about 1 mm,
and all 75 erased-cup frames rejected. On three saved frames from the live
repeat, mean branch height error dropped from 7.11 to 1.28 percentage points.
The 30 reserved fixture IDs, positions and seeds are the same in both plans.
[Anchor replay](../evidence/bottle-size-A/wide-mixed-r7b-cup-anchor-replay.json),
[revised plan](../evidence/bottle-size-A/wide-mixed-r7b-plan.json).

A second targeted repeat reached 86.55% of rim height and abstained after
the RGB estimate rose above the first correction window. The extended
settled-only fit uses 180 labeled frames across four recordings. It reduces
the last-six gap on that repeat from 4.24 to 0.92 percentage points and
preserves the archived near-full stop. The moving-pour calibration and
uncorrected near-rim guard keep their prior bounds. See the
[higher-fill replay](../evidence/bottle-size-A/wide-mixed-r7b-high-fill-replay.json).

Copy `wide-mixed-r7/src` to `wide-mixed-r7b/src` and overlay the updated
`pour_cup_outline.py`. With the first development recording available,
stage `revise_wide_mixed_r7.py` in the native study root. Run the following
serially after the first development batch has released the 4070.

```sh
python3 run_isolated_perception_job.py --helper revise_wide_mixed_r7.py --label wide-r7b-rim --seconds 300
python3 run_isolated_perception_job.py --helper extend_wide_mixed_r7_settling.py --label wide-r7b-level --seconds 300
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r7b-geometry --seconds 3600 -- --study wide-mixed-r7b --reuse-study wide-mixed-r7
python3 run_sized_autonomy.py --study wide-mixed-r7b --part development --label wide-r7b-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r7b --part test --shard "$shard" --seconds 12500 --label "wide-r7b-test-$shard" || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r7b
```

The initial [seven live repeats](../evidence/bottle-size-A/wide-mixed-r7-development-results.json)
all had zero particle loss and five strict passes. The
[revised seven repeats](../evidence/bottle-size-A/wide-mixed-r7b-development-results.json)
all met the strict task criteria with zero measured particle loss. The two
previously missed short bottles completed a 74.05%-rim empty-target pour
and a 95.58%-rim near-full pour. The corrected tall near-full repeat finished
at 90.68% of rim height. The two prior spill fixtures fully emptied, with
zero measured loss across both rounds.

The [30 fresh tests and recording audit](../evidence/bottle-size-A/wide-mixed-r7b-completion.json)
finished in 6.55 hours on the guarded RTX 4070.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 14 | 15 | 14 |
| Swapped | 15 | 11 | 14 | 11 |
| Near-full target | 15 | 14 | 15 | 14 |
| Empty target | 15 | 11 | 14 | 11 |
| All trials | 30 | 25 | 29 | 25 |

Both previously missed short-bottle layouts and both previous tall-bottle
spill layouts passed with zero particle loss at the newly shifted test
positions. All 29 pours satisfied the RGB/color agreement requirement at
stop. The largest difference was 2.77 percentage points. All 15 near-full
targets finished at 90.99–95.59% of rim height. All seven tall-bottle empty
targets reached zero source inventory, with confirmation in 0.20–0.60 s
and median 0.30 s.

A different short, narrow bottle in swapped layout 02 failed neck consensus.
Sustained cup observations were available. Four completed pours lost two
particles each. The first observed escape occurred during coarse pouring in
03 normal, 10 swapped and 14 swapped, and during settling in 11 swapped.
All 29 executed pours met the particle-tolerance score. Strict acceptance
still requires zero measured loss. The audit verified all three test reports
and recordings, the original checkpoint, 135 original sources and 2,284
protected parent files. The owned simulator exited and the 4070 was idle.

## Short narrow bottle and rim-rebound repair

The RTX 4070 fine-tuned the neck heatmap using 330 training images from
30 consumed episodes, with seven held-out episodes. The missed short,
narrow bottle passed all 20 native frames and both translated sequences,
with 0.56–0.61 mm median-consensus position errors. Across 53 native and
translated replays, the worst position error was 0.64 mm. All 740 erased
frames rejected the learned neck path. The production acceptance rule
still combines the heatmap, signed gradients and temporal consensus.
[Replay report](../evidence/bottle-size-A/wide-mixed-r8-neck-fusion.json).

All four lost pairs originated in the source bottle. Three rebounded out
near 122–129° of coarse tilt and one rebounded out near 81° as the wrist
settled after braking. Three vessel combinations now move the mouth back
9–10 mm and lower it 4–6 mm during the late pour. The fourth lowers the
mouth by 4 mm at 80–81° with a smooth return to the prior height by
84°. The original 8 mm lowering collided with the bench and cup in
clearance checks. Smaller lowerings that extended to 87° also lost
bench clearance. The planner uses
the declared geometry and the controller's own tilt phase. Measured
particles identify the repair and score the experiment offline.
[Particle traces](../evidence/bottle-size-A/wide-mixed-r8-spill-trace.json).

The [sealed plan](../evidence/bottle-size-A/wide-mixed-r8-plan.json) defines
five live repeats and fresh positions and seeds for all 15 exact swap
pairs. The recording-volume check retains 10 GiB and budgets three times
the largest recent recording per remaining trial. The 57 GiB preflight
for the earlier eight-hour collection schedule still applies to older
study runners.

Copy `wide-mixed-r7b/src` to `wide-mixed-r8/src` and overlay the changed
`pour_wide_motion.py`. Stage the checked-in helpers and the existing
`fit_wide_mixed_r6.py` in the native study root.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r7.py --label wide-r8-neck-fit --seconds 3600 -- --study wide-mixed-r8
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r8.py --label wide-r8-prepare --seconds 300
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r8-geometry --seconds 4500 -- --study wide-mixed-r8
python3 run_sized_autonomy.py --study wide-mixed-r8 --part development --label wide-r8-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r8 --part test --shard "$shard" --label "wide-r8-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r8
```

All 35 revised clearance screens and 56 native controller checks passed.
The [five Isaac Sim development repeats](../evidence/bottle-size-A/wide-mixed-r8-development-results.json)
all passed with zero measured particle loss. The short, narrow bottle
was grasped and fully emptied. The settling repair finished the near-full
pour at 93.53% of rim height. The 30 new fixtures were released after these checks.

The [30 fresh tests and recording audit](../evidence/bottle-size-A/wide-mixed-r8-completion.json)
finished in 6.73 hours on the guarded RTX 4070.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 14 | 15 | 14 |
| Swapped | 15 | 12 | 15 | 13 |
| Near-full target | 15 | 14 | 15 | 15 |
| Empty target | 15 | 12 | 15 | 12 |
| All trials | 30 | 26 | 30 | 27 |

All four repoured spill layouts passed on fresh positions. The previously
missed short, narrow bottle localized, grasped and emptied in development
and fresh testing. Three coarse pours lost particles at 118.34°, 124.56°,
and 126.94°. The remaining stop failure reached 91.40% of rim height and
the 13-pause limit. The last RGB estimate was 92.30% and the color estimate
was 93.37%; the trigger requires both settled branches to reach 92.5%.
The 14 confirmed near-full stops finished at 91.49–95.40% of rim height.
All seven tall-bottle empty targets confirmed in 0.30–0.45 s after exact
depletion, with median 0.35 s. The audit verified all 30 recordings, 135
original runtime sources and 2,284 protected files. The owned simulator
exited and the 4070 was idle.

## Three coarse-pour spills and final-pulse confirmation

The prior near-full trial reached 91.40% of measured rim height. The
size-aware camera estimates agreed, but the corrected RGB branch
remained slightly below its 92.5% stop threshold when the controller
reached the 13-pulse limit. An RTX 4070 grid fit selected a 92.19%
confirmation threshold for the 0.90× height / 0.90× diameter cup at
that limit. The decision uses the lowest of three overlapping six-frame
windows from the last eight settled RGB/color samples. Each must pass
the existing stability and agreement checks. The usual 92.5% stop
remains in force before the limit. In replay, the target's conservative
estimate was 92.37% with a measured eight-frame minimum of 91.16%.
Three training and eight archived validation episodes produced 97
eligible windows. The new threshold accepted none below 90% measured
height. [Fit and replay](../evidence/bottle-size-A/wide-mixed-r9-near-full-fit.json).

The three spill traces started during coarse pouring at 118.34°, 124.56°,
and 126.94°. The two short-bottle combinations use a 9–10 mm setback
and 5–6 mm lowering before their prior losses. The tall-bottle and small
cup use a 7 mm setback and 6 mm lowering from 113° to 117° to steer the
stream farther inside the rim at first impact. The same motion options
are checked against the table, arm and declared cup and bottle shapes.
[Spill traces](../evidence/bottle-size-A/wide-mixed-r9-spill-trace.json).

The [sealed plan](../evidence/bottle-size-A/wide-mixed-r9-plan.json) includes
four targeted development pours and 30 fresh fixtures in 15 exact swap
pairs. The stop fit and clearance helpers write only small study artifacts
and use a 12 GiB mount preflight. The wet runner checks the recording
budget for all remaining trials and retains the 10 GiB margin.

Copy `wide-mixed-r8/src` to `wide-mixed-r9/src` and overlay the revised
`pour_guard_actor.py`, `pour_size_autonomy.py` and test. Stage the
updated launchers, the two study helpers and the existing r6 and r8
helpers at the native study root.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r9_stop.py --label wide-r9-stop-fit --seconds 600 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r9.py --label wide-r9-prepare --seconds 300 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r9-geometry --seconds 4500 --minimum-free-gib 12 -- --study wide-mixed-r9
python3 run_sized_autonomy.py --study wide-mixed-r9 --part development --label wide-r9-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r9 --part test --shard "$shard" --label "wide-r9-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r9
```

The [initial live repeats](../evidence/bottle-size-A/wide-mixed-r9-development-results.json)
all had zero measured particle loss. The tall near-full repeat finished
at 91.40% and confirmed the stop. A short-bottle empty repeat exposed a
4.18-point RGB/color difference after the mouth motion changed. A second
RTX 4070 fit used 152 timestamp-aligned training frames and 160 replay
frames from two recordings of that vessel combination. The maximum
gap over the final eight held-out frames fell from 4.19 to 1.01 points
on the development recording. The correction is phase- and
geometry-specific. Braking and empty detection still have access to the
uncorrected upper level. [Color fit](../evidence/bottle-size-A/wide-mixed-r9b-draining-color-fit.json).

The [revised plan](../evidence/bottle-size-A/wide-mixed-r9b-plan.json) keeps
all 30 reserved IDs, poses and seeds. Copy the staged runtime to
`wide-mixed-r9b/src`, overlay the updated `pour_size_autonomy.py` and
stage `revise_wide_mixed_r9_color.py` at the native root. With the first
development results saved, run the following in order.

```sh
python3 run_isolated_perception_job.py --helper revise_wide_mixed_r9_color.py --label wide-r9b-color-fit --seconds 600 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r9b-geometry --seconds 4500 --minimum-free-gib 12 -- --study wide-mixed-r9b --reuse-study wide-mixed-r9
python3 run_sized_autonomy.py --study wide-mixed-r9b --part development --label wide-r9b-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r9b --part test --shard "$shard" --label "wide-r9b-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r9b
```

The [revised development runs](../evidence/bottle-size-A/wide-mixed-r9b-development-results.json)
passed all four trials with zero measured spill. The short-bottle color
gap at stop was 1.12 points. The near-full repeat confirmed at 91.09% of
rim height. All 34 geometry and 57 controller checks passed.

The [30 fresh tests and recording audit](../evidence/bottle-size-A/wide-mixed-r9b-completion.json)
finished in 6.37 hours on the guarded RTX 4070.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 13 | 14 | 13 |
| Swapped | 15 | 14 | 15 | 14 |
| Near-full target | 15 | 14 | 14 | 14 |
| Empty target | 15 | 13 | 15 | 13 |
| All trials | 30 | 27 | 29 | 27 |

The previously unconfirmed near-full layout finished at 90.94% of rim
height with zero spill and confirmed with the usual two-branch rule
before the pulse limit. The fitted final-pulse rule was exercised in
archive replay. The 14 confirmed near-full stops finished at
90.94–95.74% and the maximum level-model gap at stop across all 29
actual pours was 3.08 points. All seven tall-bottle empty targets
reached zero source inventory and confirmed in 0.30–0.60 s.

Both the tall bottle/small cup in normal 09 and the short bottle/narrow
cup in swapped 12 poured cleanly. The shifted short, narrow bottle and
wide cup in swapped 02 still lost two particles at 120.93°. A second
late coarse pair escaped at 120.24° in normal 01. Normal 12 stopped
before grasp after the bottle-neck score failed temporal consensus.
The audit verified all 30 recordings, 135 original runtime sources and
2,284 protected files. The 4070 was idle at closure.

## Normal-layout bottle localization and two late-pour rebounds

The RTX 4070 fine-tuned the RGB neck heatmap on 286 training images
from 26 episodes. Eleven whole recordings were held out of this
fine-tune, including three older localization regressions. The
selected epoch used the existing 0.70 proposal score, 0.71 gradient
score, six-pixel proposal/match limit and full temporal consensus.
Normal layout 12 recovered all 20 native camera frames at 0.109 mm
consensus error. Across 37 native and 24 translated replays, the
largest error was 1.80 mm. The learned path rejected all 740 erased
frames. [Neck replay](../evidence/bottle-size-A/wide-mixed-r10-neck-fusion.json).

Both prior losses were bottle-origin droplets during coarse pouring.
The tall, narrow bottle bounced droplets forward off the rim around
120.24°. Its path now sets the mouth back 9 mm and lowers it 4 mm
from 114° to 118°. The short, narrow bottle rebounded laterally at
120.93°. Its path now sets the mouth back 7 mm and lowers it 6 mm
from 114° to 119°. The camera and stop criteria carry forward from
the previous checkpoint. [Particle traces](../evidence/bottle-size-A/wide-mixed-r10-spill-trace.json).

The [sealed plan](../evidence/bottle-size-A/wide-mixed-r10-plan.json)
defines five development pours, including two shifted spill checks,
and fresh positions and seeds across 15 exact swap pairs.

Copy `wide-mixed-r9b/src` to `wide-mixed-r10/src` and stage the updated
launchers, `fit_wide_mixed_r7.py`, `prepare_wide_mixed_r10.py` and
their existing r6/r8 imports in the native study root.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r7.py --label wide-r10-neck-fit --seconds 3600 --minimum-free-gib 12 -- --study wide-mixed-r10
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r10.py --label wide-r10-prepare --seconds 300 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r10-geometry --seconds 4500 --minimum-free-gib 12 -- --study wide-mixed-r10
python3 run_sized_autonomy.py --study wide-mixed-r10 --part development --label wide-r10-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r10 --part test --shard "$shard" --label "wide-r10-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r10
```

The [first development batch](../evidence/bottle-size-A/wide-mixed-r10-development-results.json)
verified both spill paths at both positions with zero measured loss.
The new-seed normal-12 sequence still timed out at bottle consensus.
A second 4070 fit added that recording, used 317 training images across
27 episodes, and replayed the full 41-frame startup on both target
recordings. Each target matched 41/41 frames across its native and
two translated sequences, with consensus error at or below 0.23 mm.
The fit retained 11 held-out episodes and accepted none of 802 erased
frames. It uses the 0.70 proposal and 0.68 gradient gates for the short
standard-width bottle, with the existing six-pixel agreement and
temporal consensus. The first fit is retained for the other bottle
sizes. [Full-startup replay](../evidence/bottle-size-A/wide-mixed-r10b-neck-fusion.json).

Copy the staged runtime to `wide-mixed-r10b/src` and stage
`revise_wide_mixed_r10_neck.py`. The
[revised plan](../evidence/bottle-size-A/wide-mixed-r10b-plan.json)
preserves all 30 fresh fixture IDs, poses and seeds. The revised
geometry check reuses only hash-verified clearance results for
identical geometry, motion and seed.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r7.py --label wide-r10b-neck-fit --seconds 3600 --minimum-free-gib 12 -- --study wide-mixed-r10b
python3 run_isolated_perception_job.py --helper revise_wide_mixed_r10_neck.py --label wide-r10b-prepare --seconds 300 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r10b-geometry --seconds 4500 --minimum-free-gib 12 -- --study wide-mixed-r10b --reuse-study wide-mixed-r10
python3 run_sized_autonomy.py --study wide-mixed-r10b --part development --label wide-r10b-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r10b --part test --shard "$shard" --label "wide-r10b-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r10b
```

All five [revised development pours](../evidence/bottle-size-A/wide-mixed-r10b-development-results.json)
passed with zero measured loss. Normal layout 12 grasped and stopped
near full at 94.47%. Both spill repairs passed their original and
shifted development positions. The 35 geometry and 57 controller checks
passed.

The [30 fresh trials and recording audit](../evidence/bottle-size-A/wide-mixed-r10b-completion.json)
finished in 6.41 hours on the guarded RTX 4070.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 13 | 13 | 13 |
| Swapped | 15 | 12 | 15 | 12 |
| Near-full target | 15 | 14 | 14 | 14 |
| Empty target | 15 | 11 | 14 | 11 |
| All trials | 30 | 25 | 28 | 25 |

Both prior spill layouts passed at shifted positions with zero measured
loss. Normal 12 grasped, filled to 93.05% of rim height, and stopped
with zero measured loss. The benchmark had two other bottle-neck
consensus timeouts, in normal 07 and 14. The cup confirmed in both.
Three swapped empty-target pours lost particles during coarse pouring.
Swapped 00 lost one at 107.31°, swapped 04 lost two at 128.10°, and
swapped 06 lost two at 126.56°. All 28 verified grasps completed their
ordered returns. The audit verified the 30 recordings, 135 original
runtime sources and 2,284 protected files. The 4070 was idle at closure.

## Two bottle-neck timeouts and three swapped-layout spills

The RTX 4070 fine-tuned a fallback neck detector on 349 training
images from 29 episodes, with 13 whole recordings held out. The
first fit missed an older normal-14 recording. The accepted fit
included that earlier view and checked another archived normal-14
recording as a holdout. It passed native and translated replay
on both recorded target startups. The deployed controller checks
the incumbent heatmap first and uses the retrained fallback for
the two affected bottle shapes when the incumbent rejects.
The production sequence passed 21 native and four translated
replays with worst consensus error of 1.37 mm. On the two missed
native views, the fallback provided 41 and 40 of 41 frames,
respectively. The combined learned paths rejected all erased
frames in the scoped replays. [Detector fit](../evidence/bottle-size-A/wide-mixed-r11-neck-fusion.json),
[combined replay](../evidence/bottle-size-A/wide-mixed-r11-neck-cascade-replay.json).

All three lost groups originated in the bottle during coarse pouring.
Swapped 00 first observed the single droplet above the cup near
107.31°, with the escape traced later at about 118°. Swapped 04 and
06 lost paired droplets near 128.10° and 126.56°. Their geometry-specific
mouth trajectories apply setbacks of 6, 9 and 10 mm and lowerings of
4, 5 and 6 mm, respectively, before those loss angles.
[Particle traces](../evidence/bottle-size-A/wide-mixed-r11-spill-trace.json).

The [sealed plan](../evidence/bottle-size-A/wide-mixed-r11-plan.json)
specifies five targeted development cases and 30 fresh trials across
15 exact normal/swapped pairs. Copy `wide-mixed-r10b/src` to
`wide-mixed-r11/src`, overlay the updated `pour_guard_vision.py` and
its test, and stage the fit, preparation and launch helpers at the
native study root. The prepare helper imports the existing r6, r8
and r10 utilities and the updated neck fitting helper.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r7.py --label wide-r11-neck-fit --seconds 3600 --minimum-free-gib 12 -- --study wide-mixed-r11
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r11.py --label wide-r11-prepare --seconds 600 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r11-geometry --seconds 4500 --minimum-free-gib 12 -- --study wide-mixed-r11
python3 run_sized_autonomy.py --study wide-mixed-r11 --part development --label wide-r11-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r11 --part test --shard "$shard" --label "wide-r11-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r11
```

The [five targeted development trials](../evidence/bottle-size-A/wide-mixed-r11-development-results.json)
passed with zero loss. The fallback supplied both previously missed
bottle detections. Normal 07 emptied the bottle and normal 14 stopped
near full at 95.26% of rim height. All 35 geometry tests and 58
controller tests passed.

The [30 fresh trials and audit](../evidence/bottle-size-A/wide-mixed-r11-completion.json)
finished in 6.78 hours on the guarded RTX 4070.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 14 | 15 | 14 |
| Swapped | 15 | 15 | 15 | 15 |
| Near-full target | 15 | 15 | 15 | 15 |
| Empty target | 15 | 14 | 15 | 14 |
| All trials | 30 | 29 | 30 | 29 |

All five targeted layouts passed at their shifted positions. The
fallback confirmed the bottle in normal 07. The incumbent detector
confirmed normal 14, which filled to 93.26% of rim height. The only
observed particle loss was in normal 01, during coarse pouring at
112.11°. All 30 stopped for the expected condition and completed
ordered side returns. Near-full results ranged from 91.14% to 95.76%
of rim height. The audit verified all 30 recordings, 135 original
runtime sources and 2,284 protected files. The RTX 4070 was idle and
the RTX 4090 humanoid job was still active at closure.

## Earlier mouth adjustment for normal layout 01

The last r11 failure was one bottle-origin particle in normal 01.
At 112.11° it was outside the far rim with positive forward velocity.
The RTX 4070 optimized the motion ramp using incoming particle
positions and velocities sampled above the receiving rim. A gravity-only
projection estimated the impact radius for 108 candidate paths and
the incumbent. The objective penalized near-wall impact, rim crossings
and unnecessary movement. Three archived pours supplied 6,323 fitting
observations. Two complete held-out pours supplied 4,097 observations.
The selected 6 mm setback and 4 mm lowering ramp spans 104–109°.
Held-out 95th-percentile projected impact radii fell from 13.20–13.24 mm
to 12.22–12.25 mm. This projection measures the incoming trajectory.
Live Isaac Sim pours measure actual loss after impact and rebound.
[Trajectory fit](../evidence/bottle-size-A/wide-mixed-r12-stream-fit.json),
[escaped particle](../evidence/bottle-size-A/wide-mixed-r12-spill-trace.json).

The [sealed plan](../evidence/bottle-size-A/wide-mixed-r12-plan.json)
applies the motion update to the single affected cup/bottle geometry
in both orientations. It keeps the r11 perception and stop models.
Development covers the original normal pose, two independent normal
offsets and the swapped counterpart. The 30 fresh tests span 15 exact
swap pairs. The runner reserves twice the largest recent recording
per remaining trial, plus 10 GiB of retained space. For 30 trials that
is 26.15 GiB.

Copy `wide-mixed-r11/src` to `wide-mixed-r12/src`, overlay the
revised `pour_wide_motion.py` and `test_pour_size_autonomy.py`, and
stage the fit, preparation and run helpers in the native root. The
prepare helper imports the existing r6, r8 and r10 utilities.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r12_stream.py --label wide-r12-stream-fit --seconds 600 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r12.py --label wide-r12-prepare --seconds 300 --minimum-free-gib 12
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r12-geometry --seconds 4500 --minimum-free-gib 12 -- --study wide-mixed-r12
python3 run_sized_autonomy.py --study wide-mixed-r12 --part development --label wide-r12-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study wide-mixed-r12 --part test --shard "$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study wide-mixed-r12
```

All four [development pours](../evidence/bottle-size-A/wide-mixed-r12-development-results.json)
passed with zero measured particle loss. The three normal pours
emptied the bottle at the original and shifted positions. The swapped
pour confirmed near full at 92.45% of rim height. All 34 sampled
geometry checks and 59 controller checks passed.

After the first ten fresh tests completed, the host Docker service
restarted during the second shard. The interrupted container exited
with status 137. Trial 05 normal had already finished and its sealed
recording contains a two-particle loss. Trial 05 swapped had a partial
recording. The guarded resume validated the original container, launch
manifest, plan digests and GPU limits, recorded the interruption and
kept the finished result. It reran the incomplete trial with the same
fixture and seed and then ran the eight unstarted trials. The partial
recording remains outside the scored results.

```sh
python3 run_sized_autonomy.py --study wide-mixed-r12 --part test --shard 2 --label wide-r12-test-2-resume --seconds 12500 --resume-interrupted wide-r12-test-2
python3 run_sized_autonomy.py --study wide-mixed-r12 --part test --shard 3 --label wide-r12-test-3 --seconds 12500
python3 close_wide_mixed_autonomy.py --study wide-mixed-r12
```

The [30 fresh trials and recording audit](../evidence/bottle-size-A/wide-mixed-r12-completion.json)
accounted for 7.01 hours of simulator time across the original and
resumed containers.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 13 | 15 | 13 |
| Swapped | 15 | 13 | 15 | 13 |
| Near-full target | 15 | 15 | 15 | 15 |
| Empty target | 15 | 11 | 15 | 11 |
| All trials | 30 | 26 | 30 | 26 |

Normal 01 reached zero source inventory and confirmed the stop
0.40 s later, with zero spill. Swapped 01 reached 92.70% of rim
height with zero spill. Four other empty-target pours lost two
particles each during coarse pouring. The first observed losses
were normal 05 at 119.74°, swapped 08 at 129.21°, swapped 10 at
117.83°, and normal 11 at 121.25°. All 30 trials completed their
intended stop and side return. The 15 near-full fills settled to
91.60–94.94% of rim height. The audit verified the 30 recordings,
135 original runtime sources and 2,284 protected files. The RTX
4070 was idle and the recording volume had about 12 GiB free.

## Coarse-pour mouth paths for the four r12 spills

Normal 05 and 11 and swapped 08 and 10 spilled two bottle-origin
particles each during the coarse pour. The RTX 4070 fitted smooth
tilt-dependent setback and drop ramps. The objective combines a
gravity-only impact-radius projection over the incoming stream with
a higher weight around each observed rebound angle. Three archived
pours per geometry supplied 18,104 incoming-particle observations
for fitting. Three other pours per geometry supplied 18,016
observations for holdout. The predicted held-out impact loss
decreased for each geometry, and no recording gained a predicted
rim crossing. Actual splash and retention are scored in Isaac Sim.
[Fit receipt](../evidence/bottle-size-A/wide-mixed-r13-stream-fit.json),
[particle traces](../evidence/bottle-size-A/wide-mixed-r13-spill-trace.json).

| Layout | Ramp (°) | Setback (mm) | Drop (mm) | Held-out objective reduction |
|---|---:|---:|---:|---:|
| Normal 05 | 106–113 | 6 | 3 | 49.5% |
| Swapped 08 | 112–124 | 10 | 3 | 50.8% |
| Swapped 10 | 106–119 | 4 | 3 | 4.4% |
| Normal 11 | 108–121 | 10 | 5 | 27.4% |

The [sealed plan](../evidence/bottle-size-A/wide-mixed-r13-plan.json)
uses these settings for both orientations of each affected vessel
pair with the existing perception and stop models. It sets aside four
same-pose development pours and 30 fresh tests across 15 matched
swap pairs. Both vessel positions shift by independently signed
1.1–2.5 mm per world axis. Near-full and empty targets are each
represented by 15 fresh trials.

This round fits trajectory parameters from recorded particle positions
and velocities on CUDA. The bottle geometry and commanded tilt index
the runtime correction. The inherited behavior-cloning network and
RGB perception weights supply the existing motion initialization
and visual estimates. Their neural weights receive zero further
training epochs in this round.

Stage the two stream-fit helpers, the preparation script and the
current launch, preflight and audit scripts from `tools/` in
`/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1`.
Copy `wide-mixed-r12/src` to `wide-mixed-r13/src`. The fit reads from
the migrated recording volume. The simulator writes native
recordings to `wide-mixed-r13/recordings` on the separate Ubuntu
root partition. The runner checks current-shard capacity and
records its live free-space floor in each batch receipt. The initial
development runs used a 9 GiB root-volume floor.

```sh
cd /home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1
study=wide-mixed-r13
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r13_stream.py --label wide-r13-stream-fit --seconds 600 --minimum-free-gib 1
sed 's/wide-mixed-r12/wide-mixed-r13/' wide_mixed_r12_checks.py > wide_mixed_r13_checks.py
python3 run_isolated_perception_job.py --helper wide_mixed_r13_checks.py --label wide-r13-controller-tests --seconds 360 --minimum-free-gib 1
cp wide-r13-controller-tests/run.log "$study/controller-tests.log"
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r13.py --label wide-r13-prepare --seconds 300 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r13-geometry --seconds 5400 --minimum-free-gib 1 -- --study "$study"
python3 run_sized_autonomy.py --study "$study" --part development --label wide-r13-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study "$study" --part test --shard "$shard" --label "wide-r13-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study "$study"
```

The [first four development pours](../evidence/bottle-size-A/wide-mixed-r13-development-results.json)
emptied their bottles, with three zero-spill passes. Normal 11
lost two particles at 121.63° and the fresh trials stayed sealed.
Those particles left in the incoming stream about 7.8 mm behind
the cup center and rebounded forward and upward. The original
late forward advance cancelled much of the fitted setback at
that angle.

The [4070 refinement](../evidence/bottle-size-A/wide-mixed-r13b-stream-fit.json)
adds the traced outgoing rebound to the existing incoming stream
fit. The final search required at least 5.5 mm more horizontal
clearance and a projected rebound radius below 95% of the rim.
The normal 11 checkpoint selected a 9 mm setback, 7 mm drop
and 1 mm forward advance for about 6 mm more horizontal clearance
at the rebound angle. The projected outgoing radius went from
1.32 to 0.92 times the rim radius. Held-out incoming projections
added no rim crossings, and the 95th-percentile radii stayed
below 35% of the cup rim radius. The earlier refinement did not
pass its holdout objective gate. [First refinement receipt](../evidence/bottle-size-A/wide-mixed-r13b-first-refinement.json).

Copy the r13 runtime to `wide-mixed-r13b/src` and overlay
`runtime/src/pour_wide_motion.py`, which admits up to 8 mm
of late lowering for the geometry screen. Stage the refinement
and preparation helpers alongside the earlier helpers.
The two-second development empty-confirmation gate applies
to the two tall bottle cases. The short bottles retain their
existing confirmation and source-depletion checks. The second
test shard uses the migrated recordings volume, with a 3 GiB
live floor. The development batch and other test shards use
the study volume. The final root-volume shard uses a 7 GiB live
floor, with 384 MiB reserved per planned capture.

```sh
cd /home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1
study=wide-mixed-r13b
python3 run_isolated_perception_job.py --helper refine_wide_mixed_r13_stream.py --label wide-r13b-stream-refit --seconds 600 --minimum-free-gib 1
sed 's/wide-mixed-r12/wide-mixed-r13b/' wide_mixed_r12_checks.py > wide_mixed_r13b_checks.py
python3 run_isolated_perception_job.py --helper wide_mixed_r13b_checks.py --label wide-r13b-controller-tests --seconds 360 --minimum-free-gib 1
cp wide-r13b-controller-tests/run.log "$study/controller-tests.log"
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r13b.py --label wide-r13b-prepare --seconds 300 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r13b-geometry --seconds 5400 --minimum-free-gib 1 -- --study "$study"
python3 run_sized_autonomy.py --study "$study" --part development --label wide-r13b-dev --seconds 7200
for shard in 1 2 3; do
    python3 run_sized_autonomy.py --study "$study" --part test --shard "$shard" --label "wide-r13b-test-$shard" --seconds 12500 || break
done
python3 close_wide_mixed_autonomy.py --study "$study"
```

The repair's closed recordings are archived losslessly to conserve
space. Each archive is decoded and matched to its original HDF5
byte count and SHA256 before the original is released. The audit
streams the archive again to verify the same native identity.
The archived episodes retain every image and observation.

```sh
python3 compress_sized_recordings.py --study wide-mixed-r13b --part development --release-verified-originals
python3 compress_sized_recordings.py --study wide-mixed-r13b --part test --shard 1 --release-verified-originals
```

Repeat archival for each closed test shard. Archive manifests are
named `development-recording-archives.json` and
`test-01-recording-archives.json` through
`test-03-recording-archives.json`. Each entry gives the original
path, byte count, SHA256 and archive path. To replay or refit a raw
episode, restore that manifest entry first with
`sudo zstd -d --keep -o "$original" "$archive"` and verify the
recorded SHA256.

After the final shard was archived, the earlier closure was retained
as `pre-archive-completion.json`. The closure command ran again
against every decompressed original identity. The
[final audit](../evidence/bottle-size-A/wide-mixed-r13b-completion.json)
verified all 30 fresh recordings, all four passing development
recordings, 135 original runtime sources and 2,284 protected parent
files. The fresh benchmark took 6.75 hours on the RTX 4070.
All four targeted layouts passed their new shifted fixtures.
The policy completed all 30 grasps, intended stops and ordered
side returns. All 15 swapped trials measured zero loss.

| Fresh evaluation | Trials | Strict passes | Verified grasps | Spill-free pours |
|---|---:|---:|---:|---:|
| Normal | 15 | 13 | 15 | 13 |
| Swapped | 15 | 15 | 15 | 15 |
| Near-full target | 15 | 14 | 15 | 14 |
| Empty target | 15 | 14 | 15 | 14 |
| All trials | 30 | 28 | 30 | 28 |

The 15 near-full cups finished at 91.11–95.37% of rim height.
All seven tall empty targets confirmed within 0.60 s after
measured depletion, with median 0.35 s. Normal 00 lost two
particles while braking at 82.59°. Normal 01 lost four particles
during coarse pouring at 113.99°. Their traces are retained for
the next repair. [Normal 00 trace](../evidence/bottle-size-A/wide-mixed-r13b-00_normal-remaining-spill.json),
[normal 01 trace](../evidence/bottle-size-A/wide-mixed-r13b-01_normal-remaining-spill.json).

The four initial development recordings, four passing revised
development recordings and 30 fresh recordings total 7.42 GiB
of original HDF5 bytes. Their verified lossless archives occupy
3.39 GiB, saving 4.03 GiB. The simulator containers exited normally.
The RTX 4070 was idle at closure, and the separate RTX 4090
job remained active. The final root and migrated recording
volumes had 9.60 and 6.46 GiB free.

```sh
python3 compress_sized_recordings.py --study wide-mixed-r13b --part test --shard 3 --release-verified-originals
mv -n wide-mixed-r13b/completion.json wide-mixed-r13b/pre-archive-completion.json
test ! -e wide-mixed-r13b/completion.json
python3 close_wide_mixed_autonomy.py --study wide-mixed-r13b
```

## November 18, 2026 · Terminal braking and coarse pouring

The previous Isaac Sim benchmark lost two particles in normal 00 and four in normal 01. Tracing every final escaped particle identified two separate coarse-pour rebounds in normal 01, at approximately 114° and 119.5°. The first pair was already rebounding when it first appeared above the liquid. The second pair was falling at first observation and rebounded 50 ms later. [Particle traces](../evidence/bottle-size-A/wide-mixed-r14-spill-trace.json)

The RTX 4070 compared 21 terminal-braking settings and 65 coarse-pour settings. For each geometry the fit used three archived training trials and three separate archived validation trials. Normal 00 used 1,094 training and 1,220 validation particle observations. Its fit selected a 5 mm low-angle mouth drop and a smooth braking taper from 45°/s above 90° to 36°/s below 82°. Normal 01 used 2,863 training and 3,313 validation observations. Its fit selected a 10 mm setback and 8 mm drop, applied over 104–109°, with the late forward advance set to zero. The fit reduced the projected rim-crossing distance of both observed coarse-pour rebounds. Even with the new parameters those projections remained outside the rim. Live pours measure the actual spill. Neural-training epochs this round were zero. [CUDA fit](../evidence/bottle-size-A/wide-mixed-r14-stream-fit.json)

The frozen plan contains four development pours at the two original positions in each orientation and 30 fresh tests in 15 matched normal/swapped pairs. Fresh positions have independent 1.1–2.5 mm offsets on each axis relative to the corresponding previous wide fixture. Each pair exchanges the object positions. The simulation uses the existing nine cup sizes and four bottle shapes. Each native recording is archived losslessly, then verified against its original size and SHA256. The ten small test batches keep disk headroom on Ubuntu. The development and even-numbered test batches store their verified archives in the project's directory on the mounted `/data` volume. [Frozen manifest](../evidence/bottle-size-A/wide-mixed-r14-plan.json)

Use the existing Ubuntu study root and the same container image recorded in the preceding revision. Copy the r14 helpers and updated common runners from `tools/` to the study root before running these commands. Use a fresh study directory with no previous trial outputs.

```sh
cd /home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1
mkdir wide-mixed-r14
cp -a wide-mixed-r13b/src wide-mixed-r14/src
# Copy the repository's runtime/src/pour_guard_actor.py to wide-mixed-r14/src/.
sed 's/wide-mixed-r12/wide-mixed-r14/' wide_mixed_r12_checks.py > wide_mixed_r14_checks.py
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r14_stream.py --label wide-r14-stream-fit --seconds 600 --minimum-free-gib 1
bash run_wide_mixed_r14.sh
```

The four development pours passed strict evaluation, including both targeted cases and their swapped counterparts. All 34 planned fixtures passed geometry checks and 59 controller unit tests passed. As free system-disk space decreased, the first development launch stopped before starting a trial. The launcher picked the migrated recording volume and resumed from the verified geometry with `bash run_wide_mixed_r14.sh --from-development`. This changed only where recordings were written. [Development results](../evidence/bottle-size-A/wide-mixed-r14-development-results.json) · [Geometry checks](../evidence/bottle-size-A/wide-mixed-r14-geometry.json)

| Fresh Isaac Sim trials | Strict passes | Zero measured spill | Intended stop | Grasp and return |
|---|---:|---:|---:|---:|
| Normal positions | 13/15 | 14/15 | 14/15 | 15/15 |
| Swapped positions | 14/15 | 14/15 | 15/15 | 15/15 |
| Total | 27/30 | 28/30 | 29/30 | 30/30 |

Normal 00 stopped at 92.85% of rim height. Normal 01 emptied the bottle. Both had zero spill. Two particles escaped during coarse pouring in swapped 00, first observed at 122.95°. Two escaped during braking in normal 10 at 83.97°. Normal 04 returned the bottle with zero spill and 91.61% measured rim height. Its visual estimates at stop were 92.37% and 93.18%, and its stop reason was `fill_unconfirmed` on pause 13. The seven tall empty-target trials confirmed depletion within 0.60 s, with a 0.30 s median. Inspect the 122.95° coarse stream in swapped 00 and the 83.97° brake in normal 10. Replay the final settling window of normal 04 against its checkpoint before changing the confirmation rule. [Failure diagnostics](../evidence/bottle-size-A/wide-mixed-r14-fresh-failures.json)

The ten RTX 4070 evaluation batches took 6.87 elapsed hours in total. The separate four-trial development batch took 53.2 minutes. Personal working time was unmeasured. The final audit checked the native identities of all 30 fresh recordings, 135 original runtime sources and 2,284 protected parent files. The 30 fresh recordings occupied 5.95 GiB before lossless compression and 2.71 GiB afterward. The 4070 was idle at closure. The independent 4090 container exited with code 1 at 13:44 UTC. Docker reported no OOM kill, and the cause was undetermined. This study used only the 4070 and issued no control commands to the 4090 job. All results are from Isaac Sim. [Final audit](../evidence/bottle-size-A/wide-mixed-r14-completion.json)

To reproduce the failure evidence after closure, run `python3 audit_wide_mixed_failures.py wide-mixed-r14` from the Ubuntu study root.

## Revision 15 · Coarse stream, braking and final-pause confirmation

In the previous 30 fresh Isaac Sim pours, swapped 00 lost two particles at 122.95° and normal 10 lost two while braking at 83.97°. Normal 04 had zero spill and reached 91.61% of rim height. At its 13th and final settling pause the conservative visual estimate was 92.26%, below the 92.5% confirmation setting. [Prior failure traces](../evidence/bottle-size-A/wide-mixed-r14-fresh-failures.json)

For each affected geometry the RTX 4070 fitted motion parameters on three archived trials and validated on three separate trials. The coarse fit compared 65 candidates using 1,454 training and 1,410 validation incoming-particle observations. It chose a 7 mm setback, 8 mm drop and zero late forward advance. The first geometry scan detected bench intersections at 129–131°. The revised fit restores the previous high-angle pose smoothly from 124° to 128°, ending at a net 2 mm forward offset and 4 mm drop. The braking fit compared 21 candidates using 1,573 training and 1,600 validation observations and selected a 5 mm low-angle drop and a finish rate of 32°/s. All 34 revised fixtures passed geometry preflight. [First geometry scan](../evidence/bottle-size-A/wide-mixed-r15-first-geometry.json) · [Final geometry](../evidence/bottle-size-A/wide-mixed-r15-geometry.json) · [Motion fit](../evidence/bottle-size-A/wide-mixed-r15-stream-fit.json)

For normal 04, the 4070 fitted the existing final-pause confirmation rule to the lower of two visual level estimates over three consecutive stable windows. The fitted 91.88% cutoff applies only at the final allowed pause for this checkpoint. The ordinary 92.5% target and the simulator's independent 90% scoring threshold still apply. Six training and eight held-out trials contributed windows. The fitted rule accepted none of the six underfilled training windows or eight underfilled held-out windows. This experiment tuned motion and decision parameters with zero additional neural-training epochs. [Confirmation fit and replay](../evidence/bottle-size-A/wide-mixed-r15-near-full-fit.json)

Each of the 15 wide fixtures has a matched swapped counterpart. The fresh set offsets each placement axis by 1.1–2.5 mm from its preceding wide fixture, and covers nine cup sizes and four bottle shapes. Seeds, positions and checkpoint hashes are frozen in the manifest. [Frozen manifest](../evidence/bottle-size-A/wide-mixed-r15-plan.json)

### Reproduction on the Ubuntu workstation and Mac

Start with the Ubuntu revision 14 study and archived inputs. Stage the updated repository helpers in the existing study root and copy the updated `runtime/src/pour_wide_motion.py` into a new `wide-mixed-r15/src` copied from revision 14. Generate `wide_mixed_r15_checks.py` by substituting the new study directory in the preceding revision's check driver.

```sh
cd /home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r15.py --label wide-r15-clearance-fit --seconds 600 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper wide_mixed_r15_checks.py --label wide-r15-clearance-checks --seconds 360 --minimum-free-gib 1
cp wide-r15-clearance-checks/run.log wide-mixed-r15/controller-tests.log
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r15.py --label wide-r15-clearance-prepare --seconds 360 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r15-clearance-geometry --seconds 5400 --minimum-free-gib 1 -- --study wide-mixed-r15
python3 run_sized_autonomy.py --study wide-mixed-r15 --part development --label wide-r15-dev --seconds 7200
```

After development, run `python3 -u tools/run_wide_mixed_r15.py` from the Mac project checkout. It uses the configured `kfuUbuntu` SSH alias, checks each closed batch, and stores its durable zstd archives under ignored `data/wide-mixed-r15/`. An original is released only after the Mac verifies both compressed and decompressed SHA256 and byte counts. Ubuntu retains trial metadata and per-batch receipts. After closure, the runner releases only verified temporary archive copies from RAM. To run a later full archive audit on Ubuntu, first restage the Mac archives into the directory in each receipt.

Four of four development pours passed and all 59 controller unit tests passed. The 30 fresh attempts completed on November 18 Pacific time (November 19 at 02:46:28 UTC). Their individual container wall times sum to 6.44 hours of evaluation. Personal working time was unmeasured. [Development results](../evidence/bottle-size-A/wide-mixed-r15-development-results.json) · [Final audit](../evidence/bottle-size-A/wide-mixed-r15-completion.json)

| Fresh Isaac Sim evaluation | Attempts | Strict passes | Completed zero-spill pours | Stopped before grasp |
|---|---:|---:|---:|---:|
| Normal | 15 | 14 | 14 | 1 |
| Swapped | 15 | 15 | 15 | 0 |
| Total | 30 | 29 | 29 | 1 |

Swapped 00 emptied the bottle. Normal 10 stopped at 92.08% of rim height, and normal 04 at 93.81%. All three measured zero lost particles. All 15 empty targets passed. The 14 completed near-full pours finished at 91.17–95.42% of rim height. All seven tall-bottle empty stops confirmed within 0.60 s, with a 0.35 s median.

Normal 02 transferred 0 mL. The bottle locator had 20 votes, while the cup locator had no consensus at 2.105 simulated seconds. The controller stopped before grasp. This explains why all 30 attempts recorded zero loss but only 29 count as successful pours. Next, replay its startup camera sequence and compare the missing cup consensus with nearby passing placements. [Failure diagnostics](../evidence/bottle-size-A/wide-mixed-r15-fresh-failures.json)

The final recording audit checked 30 fresh and four development recordings, 135 original runtime sources and 2,284 protected files. The Mac runner verified each compressed archive against its original native bytes, then released Ubuntu staging RAM. The 4070 was idle at closure. The execution and monitoring scripts controlled only study-owned jobs. These results are entirely in simulation.

## Revision 16 · Startup cup localization

Normal 02 stopped before grasp in revision 15. On its 42 recorded startup frames the detector initially selected a distractor, which the presence model rejected. From frame index 6 onward, the estimated rim was within roughly one pixel of the declared center, but the heatmap confidence stayed between 0.32 and 0.38, below the checkpoint's 0.397 threshold. Two nearby passing recordings from revisions 11 and 12 confirmed by frame indices 11 and 7. [Startup diagnostics](../evidence/bottle-size-A/wide-mixed-r16-startup-diagnosis.json)

After 12 unconfirmed startup frames, a marginal proposal can now be checked against two additional pixel alignments of the same RGB image. Both views must pass the existing detector threshold, agree with the original rim within the existing 4-pixel consensus radius, and pass cup-presence verification. The original view must also pass a 0.90 presence check. The existing temporal vote, reference establishment and deadline still apply. Replay also exposed a pre-existing 2.74 mm error in swapped 07. When its coarse and fine rim estimates agreed within 2.5 pixels, the outline refinement shifted the rim by 7.07 pixels. A guard now rejects outline jumps above 6 pixels under that agreement condition for checkpoints without a separately tuned jump guard. Swapped 07 improved to 1.39 mm. [First replay](../evidence/bottle-size-A/wide-mixed-r16-shift-only-replay.json) · [Startup locator](../runtime/src/pour_cup_outline.py)

The 4070 replayed 30 revision 15 trials and two nearby historical normal 02 trials. Six translated versions of the three normal 02 sequences brought the positive total to 38. All 38 confirmed within 2.5 mm. The maximum error was 1.391 mm. Normal 02 confirmed at frame index 19 with 0.216 mm error. Two synthetic cup-erasure controls per native sequence ran for the full 42-frame startup window, with zero acceptances across 64 sequences. The source-pinned comparison verified 140 unchanged runtime files and 14 unchanged checkpoints. The single edited runtime file implements startup cup localization. All neural weights, grasp, pour, braking, stop and return settings retain their revision 15 identities. [Replay audit](../evidence/bottle-size-A/wide-mixed-r16-cup-startup-replay.json)

The targeted evaluation uses the original failed placement. The fresh evaluation uses 15 paired wide normal/swapped layouts across nine cup sizes and four bottle shapes. Each object shifts by 1.1–2.5 mm per table axis from its corresponding revision 15 position. The plan fixes seeds and positions before autonomous evaluation. Recordings use two-trial batches and byte-verified Mac backup. [Frozen fixtures](../evidence/bottle-size-A/wide-mixed-r16-plan.json)

To reproduce, use the existing Ubuntu study root and the revision 15 checkpoints. Copy `wide-mixed-r15/src` into a new `wide-mixed-r16/src` and install the revised `pour_cup_outline.py` there. Stage the updated repository helpers in the study root. On the Mac, `.venv/bin/python tools/extract_r16_startup.py` extracts original JPEG bytes from the hashed revision 15 archives. Copy the resulting `data/wide-mixed-r16/startup.h5` to the Ubuntu study's `wide-mixed-r16/` directory. The two older recordings remain on Ubuntu at their original paths.

```sh
cd /home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1
sed 's/wide-mixed-r15/wide-mixed-r16/' wide_mixed_r15_checks.py > wide_mixed_r16_checks.py
python3 run_isolated_perception_job.py --helper diagnose_r16_startup.py --label wide-r16-diagnose --seconds 600 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper replay_r16_startup.py --label wide-r16-outline-replay --seconds 1800 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper wide_mixed_r16_checks.py --label wide-r16-checks --seconds 360 --minimum-free-gib 1
cp wide-r16-checks/run.log wide-mixed-r16/controller-tests.log
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r16.py --label wide-r16-prepare --seconds 360 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r16-geometry --seconds 5400 --minimum-free-gib 1 -- --study wide-mixed-r16
python3 run_sized_autonomy.py --study wide-mixed-r16 --part development --label wide-r16-target --seconds 7200
```

From the Mac project checkout, run `python3 -u tools/run_wide_mixed_r15.py --study wide-mixed-r16` after the targeted pour passes. This runs all 30 fresh tests and completes the Ubuntu and Mac recording audits. Durable native archives are under the Mac project's `data/wide-mixed-r16/` directory.

The targeted normal 02 pour transferred 34.01 mL, stopped near-full at 93.54% and returned with zero loss. The fresh shifted normal 02 trial confirmed both vessels after 0.60 simulated seconds. Its cup estimate was 0.27 mm from the declared fixture position and its final height was 93.58%, again with zero loss. All 30 trials confirmed both vessels. The maximum fresh cup-position error was 1.72 mm. [Targeted pour](../evidence/bottle-size-A/wide-mixed-r16-development-results.json) · [Fresh startup audit](../evidence/bottle-size-A/wide-mixed-r16-fresh-startup.json)

| Fresh Isaac Sim fixtures | Attempts | Both vessels localized | Actual pours | Strict zero-spill passes |
|---|---:|---:|---:|---:|
| Normal | 15 | 15 | 15 | 14 |
| Swapped | 15 | 15 | 14 | 12 |
| Total | 30 | 30 | 29 | 26 |

All 29 transfers reached the intended stop and returned the bottle. Normal 05 lost one particle during coarse pouring at 109.60°. Swapped 10 and swapped 14 each lost two during coarse pouring at 127.07° and 128.22°, respectively. Swapped 05 recorded both vessel locations and a simulator-verified grasp, but the controller stopped after two attempts with `grasp_unconfirmed` and transferred 0 mL. The zero-loss count is 27/30 attempts, including that no-pour attempt. Of the actual pours, 26/29 had zero loss. Under the existing particle tolerance, 29/30 attempts passed. All seven tall-bottle empty targets confirmed depletion within 0.45 s. [Failure traces](../evidence/bottle-size-A/wide-mixed-r16-fresh-failures.json) · [Completion audit](../evidence/bottle-size-A/wide-mixed-r16-completion.json)

The 15 RTX 4070 test batches took 6.60 elapsed hours in total. This was evaluation time, with zero neural-training epochs. Personal working time was unmeasured. The Mac verified all 31 native recordings, and Ubuntu released the temporary recording copies from RAM. For independent startup measurement after closure, run `python3 audit_startup_localization.py wide-mixed-r16` once from the Ubuntu study root. No real-robot trials were run.

### Revision 17 grasp confirmation and coarse pouring

The RTX 4070 fitted three geometry-specific mouth paths and a post-lift visual confirmation threshold. The candidates use the same 141 runtime files, all neural weights, both startup localizers and the same stop logic as revision 16. The post-lift check accepts a neck position within 16.44 pixels of the expected lifted neck, up from 15.69. It still needs six visual votes plus lifted-arm and obstructed-finger evidence. Both missed attempts confirmed on replay. All 51 single-evidence ablations rejected. The three mouth paths were fitted on nine archived Isaac Sim pours and checked on nine separate archived pours using a ballistic rebound proxy. Escaped-particle traces and simulator state are offline scoring inputs.

The first collision screen found eight paths with hand/table intersections late in the tilt. Limiting terminal lowering to 2 mm for the two late rebounds produced a clean 34/34 geometry screen. All four original fixtures then passed the autonomous development gate with zero lost particles. The 30 reserved placements use 15 matched normal/swapped pairs with independent 1.1–2.5 mm perturbations of the predecessor layout, across nine cup and four bottle geometries.

Reproduce from the Ubuntu study root with the parent recordings, checkpoints, licensed Isaac Sim assets and project launchers already installed. Copy the fitting, preparation, preflight, archive and closure helpers from this checkout. In a new study directory, copy revision 16's src directory intact. Copy the revision 16 Mac archives ending in 05_normal, 10_swap and 14_swap into the inputs directory. The fitter verifies their hashes against the parent evaluation receipts. Generate wide_mixed_r17_checks.py by substituting wide-mixed-r17 for wide-mixed-r16 in wide_mixed_r16_checks.py. Use unused job labels for each invocation.

```sh
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r17.py --label wide-r17-fit-b --seconds 1800 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r17.py --label wide-r17-plan-b --seconds 600 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper wide_mixed_r17_checks.py --label wide-r17-checks --seconds 600 --minimum-free-gib 1
cp wide-r17-checks/run.log wide-mixed-r17/controller-tests.log
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r17-geometry-b --seconds 5400 --minimum-free-gib 1 -- --study wide-mixed-r17
python3 run_sized_autonomy.py --study wide-mixed-r17 --part development --label wide-r17-dev --seconds 7200
```

From the Mac project checkout, run `.venv/bin/python -u tools/run_wide_mixed_r15.py --study wide-mixed-r17`. It checks development, runs the fresh pairs and verifies the decompressed native recording hash before releasing any Ubuntu original. After closure, run `python3 audit_startup_localization.py wide-mixed-r17` in the Ubuntu study root.

The fresh run closed November 19 Pacific time with 29/30 strict zero-spill passes. All 30 transferred liquid, reached the intended stop and completed the return. Normal layouts passed 15/15 and swapped layouts 14/15. Near-full targets passed 15/15 and empty targets 14/15. Fresh swapped 05 confirmed the first lift and stopped at 94.42% of rim height. On the fresh empty targets, normal 05 and swapped 10 and 14 retained every particle. Swapped 02 lost two particles, both first seen outside the vessels at 24.705 simulated seconds and 126.43° during coarse pouring. All seven tall empty targets confirmed in 0.90 simulated seconds or less after depletion. Localization confirmed both vessels 30/30, with a maximum declared cup-position error of 2.40 mm. Fresh evaluation took 6.82 elapsed GPU-job hours. Personal effort was unmeasured. The 34 development and test recordings passed their native-content hash checks. [Completion](../evidence/bottle-size-A/wide-mixed-r17-completion.json) · [Localization](../evidence/bottle-size-A/wide-mixed-r17-fresh-startup.json) · [Particle traces](../evidence/bottle-size-A/wide-mixed-r17-fresh-failures.json)

### Revision 18 late-tilt repair for swapped 02

Two particles in revision 17 first appeared in flight at 126.43° and were next observed moving outward above the 26.4 mm rim radius. A CUDA grid search on the RTX 4070 evaluated 568 terminal poses including the incumbent. It used 1,264 incoming particle tracks in three archived pours, the two recorded exits, and 1,348 incoming tracks in three separate validation pours. The selection first required a projected radial clearance of 1.5 mm and minimized terminal displacement among candidates that achieved it. The chosen pose starts at 124° and finishes at 127°. Its offset changes from +1 to −8 mm and lowering from 6 to 8 mm. The projected rim ratio fell from 1.38 to 0.93 for both escaping particles. This ballistic approximation was tested with new Isaac Sim pours.

One field changed in the shared 02 checkpoint. The other 13 checkpoints, all trained tensor values, grasp confirmation, localization and stop logic were frozen. All 32 paths passed geometry screening. The targeted swapped 02 and normal 02 pours each grasped, transferred liquid, reached the intended stop and completed return and release with zero measured spill. The normal near-full trial settled at 93.71% of rim height.

Use the Ubuntu study root and the same Isaac Sim image, assets, source tree, isolated launcher and storage guard as revision 17. Stage the tracked helpers from this checkout and the verified Mac archive for revision 17 swapped 02 under the study inputs. In a new study directory run the following commands.

```sh
mkdir -p wide-mixed-r18/inputs
cp -a wide-mixed-r17/src wide-mixed-r18/src
sed s/wide-mixed-r17/wide-mixed-r18/ wide_mixed_r17_checks.py > wide_mixed_r18_checks.py
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r18.py --label wide-r18-fit --seconds 1800 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r18.py --label wide-r18-plan --seconds 600 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper wide_mixed_r18_checks.py --label wide-r18-checks --seconds 600 --minimum-free-gib 1
cp wide-r18-checks/run.log wide-mixed-r18/controller-tests.log
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r18-geometry --seconds 5400 --minimum-free-gib 1 -- --study wide-mixed-r18
python3 run_sized_autonomy.py --study wide-mixed-r18 --part development --label wide-r18-dev --seconds 7200
```

On the Mac, run `.venv/bin/python -u tools/run_wide_mixed_r15.py --study wide-mixed-r18` from the checkout after both targeted pours pass. The sealed evaluation uses one trial per recording batch to respect available storage and the live 3 GiB reserve. Run `python3 audit_startup_localization.py wide-mixed-r18` in the Ubuntu study root after closure.

The fresh RTX 4070 benchmark closed on November 20 with 27/30 strict zero-spill passes. Swapped 02 delivered 28.00 mL and stopped on empty. Normal 02 delivered 34.01 mL and stopped near full at 93.68% of rim height. Both completed the return with zero loss. Normal 01 confirmed the cup but could not confirm the bottle before grasp. Of the 29 actual pours, all reached the intended stop and completed the return, and 27 measured zero loss. Swapped 06 lost one particle at 111.93° and swapped 12 lost two at 129.44°, all during coarse pouring. All three failures used checkpoints with identical parent hashes. The fresh placements differ across revisions. All 15 near-full targets passed, along with 12 of 15 empty targets. The 30 batches took 6.56 elapsed evaluation hours. This revision used parameter fitting and zero neural-training epochs. Personal work time was unmeasured. [Fresh audit](../evidence/bottle-size-A/wide-mixed-r18-completion.json) · [Startup audit](../evidence/bottle-size-A/wide-mixed-r18-fresh-startup.json) · [Particle and localization traces](../evidence/bottle-size-A/wide-mixed-r18-fresh-failures.json)

The Mac verified all 32 recordings against their native-content SHA-256 before the Ubuntu duplicates were released. Three closed revision 17 replay-input copies (278,515,144 bytes total) were relocated to the training partition after content verification. Relative links preserve their original paths. The Ubuntu system volume had about 0.43 GB available at closure.

### Revision 19 bottle localization and two coarse-pour trajectories

Revision 18 normal 01 detected a neck near its calibrated projection but rejected the frame-level confidence. Revision 19 keeps the primary locator and adds a fallback for that vessel checkpoint. It reuses the neural proposal and matches a gradient template from four startup frames in each of two normal trials. The minimum proposal score is 0.53, the minimum template score is 0.80 and the two points must agree within four pixels. Existing temporal voting confirms the position. The revised check confirmed all 36 native and translated sequences from 12 recorded trials within 0.77 mm, and rejected all 12 bottle-erased sequences. The replay includes the two template trials. Simulator poses supply offline labels.

The two motion fits use observed escaped particles and gravity-based rim-crossing projections. For each layout, three recorded pours supply the training tracks and three separate recordings check the incoming stream. Swapped 06 starts setting the mouth back at 106° and blends into the original terminal offset above 118°. Swapped 12 shifts the mouth from −1 to −10 mm at 126–129° with the existing 6 mm lowering. The proxy reduced risk for both, though it still projected particles outside the rim. The targeted Isaac Sim result is the measured spill test. All 33 planned paths cleared the geometry screen. All three development trials grasped on the first attempt, delivered 28.00 mL, confirmed empty and returned with zero loss. The 02 checkpoint, grasp verification, stop logic, and 141 runtime source files are frozen.

On the Mac, run `.venv/bin/python tools/extract_r19_neck.py` against the verified revision 15–18 archives. In a new Ubuntu study directory, copy the complete revision 18 src directory to wide-mixed-r19/src. Stage neck-startup.h5 and the revision 18 archives for swapped 06 and 12 in wide-mixed-r19/inputs. Use the same Ubuntu study root, Isaac Sim image and assets as revision 18. Stage the tracked helpers from this checkout before running these commands.

```sh
sed s/wide-mixed-r18/wide-mixed-r19/ wide_mixed_r18_checks.py > wide_mixed_r19_checks.py
python3 run_isolated_perception_job.py --helper fit_wide_mixed_r19.py --label wide-r19-fit-b --seconds 1800 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper prepare_wide_mixed_r19.py --label wide-r19-plan --seconds 600 --minimum-free-gib 1
python3 run_isolated_perception_job.py --helper wide_mixed_r19_checks.py --label wide-r19-checks --seconds 600 --minimum-free-gib 1
cp wide-r19-checks/run.log wide-mixed-r19/controller-tests.log
python3 run_isolated_perception_job.py --helper preflight_sized_autonomy.py --label wide-r19-geometry --seconds 5400 --minimum-free-gib 1 -- --study wide-mixed-r19
python3 run_sized_autonomy.py --study wide-mixed-r19 --part development --label wide-r19-dev --seconds 7200
```

After all three targeted pours pass, run `.venv/bin/python -u tools/run_wide_mixed_r15.py --study wide-mixed-r19` from the Mac checkout. Each test runs in its own guarded batch. The launcher budgets native recordings on the system disk with a live 4 GiB floor, or uses the training partition with a 3 GiB floor when the system disk lacks room. The Mac verifies the original recording bytes before the Ubuntu duplicates are released. After closure run `python3 audit_startup_localization.py wide-mixed-r19` in the Ubuntu study root.

The 24 revision 13b archives previously on the system disk are in data/wide-mixed-r13b on the Mac. The immutable batch receipts still contain the original Ubuntu paths. From the Mac checkout run `.venv/bin/python tools/restore_r13b_archives.py` to verify all copies or add `--restore ARCHIVE_BASENAME` to restore one at its recorded path. A byte-verified round trip was completed before releasing the 2,301,361,893 archived bytes. [Restore inventory](../evidence/bottle-size-A/wide-mixed-r13b-mac-offload.json)
