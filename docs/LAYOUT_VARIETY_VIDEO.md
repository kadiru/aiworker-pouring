# Twelve larger layout tests

Study `A-layout-variety-20261105-v1`, prefix `V12_A_`. The user chose autonomous tests of the current learned controller, including honest failure labels. This is a deliberately difficult layout stress test, not a retraining run or a representative random success-rate estimate.

Twelve distinct layouts include four exact bottle/cup swap pairs, both diagonals, side-by-side placements, both objects nearer/farther from the robot, and either end of the table. Object centers span world X 0.40–0.58 m and Y −0.38 to −0.04 m. The same lightly frosted material A, red liquid, 1.8 mm particles, 400 Hz physics and 20 Hz controller are used. Targets alternate six near-full cups and six empty bottles. Source fills are 94/30 nominal mL, cup starts at 6 nominal mL.

The existing v18 checkpoint `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b` and actor are unchanged. There is no new fitting, threshold relaxation, teacher fallback or repeated attempt. Task/wrist calibration is unchanged; a third fixed overview camera is recorded for the video only. All twelve outcomes will appear, including failures before grasping. Zero spill from an untouched bottle is not a pass.

Before freezing, CPU checks require vessel support, inverse-kinematics reach, a raised approach clear of the worktop, and no initial robot/prop convex-envelope intersection. These checks do not prove dynamic grasping, self-collision avoidance or learned motion generalization. Source hashes and the plan are frozen before queuing. The previous W30 benchmark and its videos are preserved.

Execution is confined to the existing project-owned 4070 container: CPU quota four, CPUs 16–23, shares 256, nice 10, 16 GiB RAM/no extra swap, and immutable-ID GPU-priority guard. Only that idle container is restarted to load the recording harness. Other jobs are not signaled or reconfigured. Rendering the final labeled video uses the Mac CPU.

Current status: preparation and geometric preflight complete; native recordings and outcomes pending.
