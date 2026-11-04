# Expanded-position material-A study

Status: reference demonstrations running; no expanded-position autonomous result yet.

The user authorized larger bottle/cup position variation, retraining on the RTX 4070, thirty fresh autonomous tests aiming for zero spill, and committing/pushing. This study is separate from the completed 30/30 narrow-workspace evaluation.

## Domain and separation

Study `A-workspace-20261104-v1`, prefix `W30_A_`, reserved test seed `20261224`.

| Coordinate | Previous half-range | New half-range |
|---|---:|---:|
| Cup world X/Y around (0.49, -0.15) m | ±8 mm each | ±24 mm each |
| Bottle cup-frame X around -0.12 m | ±9 mm | ±27 mm |
| Bottle cup-frame Y around 0 m | ±4 mm | ±12 mm |

The bottle position is relative to the cup; these four independent variations also change its absolute world position. All twenty reference demos, eight development scenes and thirty tests are distinct. Every development/test scene exceeds the old half-range by at least 25% in one coordinate. Material A, fixed task/wrist cameras, vessel geometry, 1.8 mm particles, 400 Hz physics, 20 Hz observations, fill ranges and the 40-second autonomous deadline remain unchanged.

Sixteen reference demos cover all corners of the four-dimensional domain, balanced for full/empty goals along each dimension. Four separate interior demos supply validation. Collection uses the privileged scripted joint-drive reference controller, with a 5°/s slow tilt and nominal-volume stop target .88. Reference demos are training evidence, never autonomous successes. All attempts, including rejected demos, must be retained.

CPU preflight solved all wrist paths and confirmed initial bottle/cup matching-patch coverage for **58/58 scenes**. This establishes kinematic reachability and projection coverage, not collision-free motion or successful grasps. [Preflight](../evidence/workspace-A/workspace-preflight.json), [plan](../evidence/workspace-A/frozen-plan.json).

## Intended fit and evaluation

The new candidate will retain the previous neural encoders and feedback controller while fitting expanded bottle/cup template families, the settled-height ridge head, and the contextual pouring movement primitive. Template search bounds must extend every retained family, not only the outer wrapper. New images and joint targets come from reference demonstrations; the completed C30 batch becomes consumed development evidence. W30 development and test trajectories cannot enter this initial fit.

Method: supervised regression / behavioral cloning plus visual servo and feedback control. Settled height minimizes episode-weighted squared prediction error plus ridge regularization. The motion model minimizes joint-target imitation error conditioned on measured lift joints and localized cup position. Simulator poses/particles supply offline labels and evaluation scores; runtime control receives RGB, joints, own history and fixed calibration.

Candidate selection uses development evidence. Only afterward are the checkpoint, source hashes and plan frozen and the thirty reserved cases run once, with fifteen full and fifteen empty targets. All failures remain in the denominator. No tuning, retries, altered thresholds or teacher fallback are permitted during that batch.

The unchanged strict goal requires verified sustained grasp, liquid transfer, correct stop reason, full upright return/release, at least 90% final rim height or an empty source, zero final outside particles and zero sampled near-table loss. Transient outside readings are reported separately. The 20 Hz loss observations do not exclude all splashes or between-sample excursions. Nominal mL remains an uncalibrated particle-count quantity; the legacy nominal-volume score stays separate.

## Resources and artifacts

Only owned container `697166fd9ca8f803f339097137f1d47fc80ff560d43bf71f0ab6d77869474cbd` was restarted, after verifying it was idle. Resource limits remain 4070-only visibility, four-core CPU quota on CPUs 16–23, CPU shares 256, nice 10, 16 GiB RAM/no extra swap, and the immutable-ID priority guard. No other job was signaled or reconfigured. [Starting snapshot](../evidence/workspace-A/runtime-before-work.json).

Native study: `/workspace/learning/A-workspace-20261104-v1` in the owned container; host base `/home/kadir/aiworker-pouring-4070`. Local small reports: `evidence/workspace-A/`; raw backups will use `data/material-A-workspace/`. Raw recordings, model weights, images and videos stay outside Git. Preserve the previous model and all previous reports.

Final result, training counts, model/plan/source identities, every recording audit, camera review and before/after resource comparison will be added after completion. Do not present this plan or the reference collection as a passing autonomous benchmark.
