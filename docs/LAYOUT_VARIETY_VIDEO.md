# Fourteen larger layout tests

Study `A-layout-variety-20261105-v1`, prefix `V12_A_`. The user chose autonomous tests of the current learned controller, including honest failure labels. This is a deliberately difficult layout stress test, not a retraining run or a representative random success-rate estimate.

Twelve distinct layouts include four exact bottle/cup swap pairs, both diagonals, side-by-side placements, both objects nearer/farther from the robot, and either end of the table. Object centers span world X 0.40–0.58 m and Y −0.38 to −0.04 m. The same lightly frosted material A, red liquid, 1.8 mm particles, 400 Hz physics and 20 Hz controller are used. Targets alternate six near-full cups and six empty bottles. Source fills are 94/30 nominal mL, cup starts at 6 nominal mL.

The existing v18 checkpoint `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b` and actor are unchanged. There is no new fitting, threshold relaxation, teacher fallback or repeated attempt. Task/wrist calibration is unchanged; a third fixed overview camera is recorded for the video only. All twelve outcomes will appear, including failures before grasping. Zero spill from an untouched bottle is not a pass.

Before freezing, CPU checks require vessel support, inverse-kinematics reach, a raised approach clear of the worktop, and no initial robot/prop convex-envelope intersection. These checks do not prove dynamic grasping, self-collision avoidance or learned motion generalization. Source hashes and the plan are frozen before queuing. The previous W30 benchmark and its videos are preserved.

Execution is confined to the existing project-owned 4070 container: CPU quota four, CPUs 16–23, shares 256, nice 10, 16 GiB RAM/no extra swap, and immutable-ID GPU-priority guard. Only that idle container is restarted to load the recording harness. Other jobs are not signaled or reconfigured. Rendering the final labeled video uses the Mac CPU.

## Preflight correction and two additional cases

The original geometric helper covered eight arm meshes but omitted four right finger meshes. Test 02 had native finger contact and spill at the first measurement. A post-hoc audit including the fingers flags 02 (bottle), 03 (cup) and 08 (bottle); the cup flag is a conservative closed-envelope intersection, not an instrumented native cup-contact measurement. Those scenes and all original outcomes remain preserved, with explicit setup labels. The original preflight is not a complete collision-clearance result.

Two additional once-only layouts, `VX2_A_test_12/13`, live in `A-layout-variety-extra-20261105-v1`. They exchange cup/bottle centers at (0.60,−0.21) and (0.52,−0.35) m. Both pass IK, a raised approach and initial prop checks using all twelve arm/finger collision meshes. They preserve the same actor, model, camera calibration and physics; only an extra queue command is added after the first twelve close and are audited. No failure is replaced or rerun.

The video therefore contains 14 attempts, five exact swap pairs and 11 starting layouts without the geometry flags. Centers span 20 cm in world X and 34 cm in Y. Seven full / seven empty targets. Deliberately selected scenes do not estimate a random-layout success rate.

## Completed recordings and video

All 14 attempts are complete. One familiar-layout pour passed at 93.197% final rim height with zero loss. The other 13 rejected localization, including the three setup flags. Test 02 lost 0.548208 nominal mL; 08 lost 43.629192. Both had native bottle/finger contact at the first measured sample. Test 03's cup overlap is a geometry flag; native cup contact was not instrumented. The two extra tests had no setup flags and no spill, but also failed localization. Twelve runs had zero loss, including eleven no-pour aborts; those aborts are not passes.

Both batches use checkpoint `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b`. The first 12 were frozen at source c65f755, extras at ab3b119. Only the queue harness changed between them. All 14 HDF5 recordings and 70 sidecars match native SHA256 values. Their 1,704 transitions pass action/timestamp/image continuity and sampled particle-inventory checks; maximum camera/state skew 6.092e-7 s. Raw files are local under `data/material-A-layout-variety/` and on Ubuntu under `outputs/fountain/20261105T215110Z/` and `outputs/fountain/20261105T222348Z/`.

The final video `evidence/layout-variety-A/fourteen-autonomous-layouts.mp4` is 120.6 s, 1280×720, H.264/yuv420p, 20 fps, 2,412 frames and 3,234,525 bytes. SHA256 `0abe8f97e266038ae04208deb666e08c26ec1e0ad3452156ff93f0ed376d397a`. Every complete attempt is retained at labeled 1×/2× speed, plus initial/final holds. Setup failures use 1×. No internal cuts, interpolation or synthetic simulation imagery. The overview, task and wrist cameras are shown with a planned-position map, final outcome and concise cause. Full decode and 44 encoded review frames covering every case passed visual QA.

Reports: [all fourteen](../evidence/layout-variety-A/final-benchmark.json), [integrity](../evidence/layout-variety-A/recording-integrity.json), [video mapping](../evidence/layout-variety-A/video-manifest.json), [encoded review](../evidence/layout-variety-A/video-review.json). This batch demonstrates the current localization limit; it does not establish successful pouring at the larger positions. No retraining occurred.

Reproduction, from the project root:

```sh
.venv/bin/python tools/combine_layout_evidence.py
.venv/bin/python tools/render_layout_variety.py
.venv/bin/python tools/review_layout_video.py
```

Native audit for the extra batch, inside the owned container: `/isaac-sim/python.sh /workspace/src/audit_layout_variety.py --extra`. To recompute the original audit after the queue-harness update, use `/isaac-sim/python.sh /workspace/src/audit_layout_variety.py --source-root /workspace/learning/A-layout-variety-extra-20261105-v1/harness-before`; its original 31 runtime sources were archived and verified before replacement. These audit commands read recordings and write reports; they do not queue trials. Media, raw data and model weights stay outside Git.
