# Wider and swapped table placements

Status: in progress, November 6, 2026, America/Los_Angeles. No new trained-candidate or reserved-test result yet. Preserve earlier W30 (23/30) and broader-layout (1/14) evidence.

## Task and evaluation

AIWorker's right arm must locate a bottle and cup from RGB, grasp and lift the free dynamic bottle, pour a red water-like drink, stop at at least 90% measured settled rim height or an empty source, return upright, place/release and retreat. Material A and native particle/vessel physics are retained. The robot receives task/wrist RGB, joint positions/velocities, its own history and fixed calibration. Simulator poses, contacts and particle counts supply training labels and independent scoring, never autonomous control inputs.

Native root: `/workspace/learning/A-wide-20261105-v1` in the existing `aiworker_pouring_4070` container, mounted from `/home/kadir/aiworker-pouring-4070`. Local raw backups: `data/material-A-wide/`; small committed reports: `evidence/wide-A/`.

The geometry-screened plan has 24 training and eight validation scripted references; eight separate development scenes; and 30 fresh reserved autonomous tests arranged in 15 exact swaps. Five spatial families include transverse/diagonal placements, with independent ±12 mm jitter and varied initial liquid levels. Each swap pair has one full-cup and one empty-source target. All layouts are distinct across splits. Tests are once-only after selection; every failure and partial attempt is retained. No teacher fallback, test retries or test-based refitting is permitted within this benchmark.

The new task lens is 30 mm (previously 40 mm), fixed 384×384 task/wrist/overview recording at 20 Hz, physics at 400 Hz. Overview is recording-only. Calibration is saved per recording. Exact vessel positions and fills are in the frozen plan, not inferred from video labels.

Primary pass requires verified physical and visual grasp, two-finger contact on at least 80% of sampled tipping frames, at least 1 nominal mL transferred, correct stop reason, completed upright return/release/retreat, no existing scoring gates, and the actual height/empty goal. Empty allowance is max(0.25 nominal mL, 1% initial source). Zero measured loss requires both final outside inventory (including airborne particles) and ever-near-worktop inventory to be zero. One particle is 0.005832 nominal mL. This sampled simulation criterion permits transient returning droplets and does not establish continuously splash-free or physical-robot performance.

## Learning and engineered components

This is supervised imitation/regression with a programmed feedback controller, not reinforcement learning or end-to-end task learning.

- Localization: reference RGB gradient template banks, view-local support, confidence calibration and temporal consensus. A partial-reference CPU diagnostic found a 23 mm bottle error where robot/background edges contaminated the original neck patch. All twelve narrowed-neck candidates on these 16 training references stayed below 0.9 mm initial error; all four original-mask candidates failed. Eight production checks reproduce the diagnostic. Narrower masks enter the candidate search; all final choices still require the complete validation set. Candidate settings are selected using reference/validation data; synthetic missing-object occlusions are explicitly synthetic checks.
- Settled height: frozen visual features and an episode-balanced ridge readout. Objective is weighted squared height error plus a quadratic coefficient penalty, `sum_i w_i (f(x_i)-h_i)^2 + lambda ||beta||²`. Near-full examples carry extra weight. New height labels use stable-inventory frames after transfer away from the cup; an upright bottle above it can occlude its surface. The stopping margin accounts separately for wider-view validation overestimation, so the larger historical dataset cannot hide a wide-view bias.
- Motion: contextual joint-target behavioral cloning from successful scripted pours. Context comprises seven measured lift joints, RGB-localized cup XY and an intercept. Targets are learned on a 166-angle grid by ridge least squares. A programmed IK projection enforces calibrated mouth geometry; the projected target is not claimed to be wholly learned.
- Empty recognition: a 65-parameter logistic classification readout on frozen RGB features, fitted with episode/class-balanced binary cross entropy plus an L2 penalty around the parent weights. The wider pilot framing probe missed the swapped empty state, so reusing all old classifier weights is insufficient. Source presence, source/wrist encoders and the other output rows remain frozen. Validation must reject false empty detections and cover every empty validation episode before deployment.
- Grasp/stop/return: visual lift consistency plus finger obstruction, settled-height feedback and irreversible stop/return states. Programmed ready/approach clearance and joint limits remain explicit.

The candidate's final frame/episode counts, selected coefficients, errors, checkpoint hashes and fit time must come from the completed model summary; they are not known yet. Frozen parent: v18 SHA256 `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b`.

## Pilot failures and clearance correction

Initial four reference pilots are all preserved. 00 reached 93.114% rim height with zero measured loss; swapped 01 emptied with zero measured loss. 02 contacted a finger at initialization and lost 27.999432 nominal mL. 03 grasped/poured but lost 0.268272 nominal mL. The original successful references are 00/01; later explicitly named repairs add accepted supervision without deleting originals.

The first preflight included all twelve arm/palm/finger meshes but incorrectly screened against the inner fluid profile. The corrected envelope adds actual 2.4 mm walls and a combined 2 mm contact margin. It identifies the two affected starts. The candidate uses a fixed ready TCP 10 cm higher and level overhead translation before descent. Geometry audits cover initial vessel contact and 41 approach samples; they exclude native tracking dynamics and general robot self-collision. The raised-ready approach passed all 70 geometry checks. Repaired 02 passed; repaired 03 still lost 28 particles during withdrawal. An over-cup upright return and drain pause then produced a zero-loss 03 repair at 93.403% rim height, permitting general collection. Reference 04 subsequently lost one particle; it is retained as a failure.

Repair identities `T30_A_demo_train_02_ready_r1`, `03_ready_r1` and `03_return_r2` map to the original planned layouts. Repaired records do not erase failed originals. Training acceptance checks require closed recordings, verified grasp, zero measured loss and no scoring gates.

New scripted empty-source references hold the verified emptied bottle at its current tilt for 1.2 seconds before returning, producing stable empty-state images for supervision. The hold never applies to full-cup stops or autonomous evaluation; both object motion and liquid remain native physics.

A further candidate retains the early lift boost until 45 degrees, adds 4 mm lip clearance and moves the final upright drip position to the cup center. Native swapped-pilot logs showed real cup-wall contact during early tilt. Candidate piecewise vessel envelopes pass all 70 sampled paths; dynamic validation is pending. This candidate is staged, not deployed into the active collection. Late one-particle losses in references 04/12 originate from the bottle during upright transfer (cup inventory unchanged). The refined candidate lengthens the over-cup drain pause from 0.4 to 1.2 s and return translation from 1.0 to 2.0 s. Its geometry audit is separate; wet validation remains pending. Geometry constraints and return timing are explicitly engineered components.

## Fresh identities and source freezing

The initial plan used the historical `T30_A_` prefix. Before any new development/test was run, an existing historical T30 benchmark was found. New development/tests use unused **`XW30_A_`** identities; the in-flight references retain `T30_A_demo_*` with protocol 17. The original plan, the 38-name mapping and corrected plan are preserved. Only identifiers changed; all numeric fixture values, splits and seeds are unchanged. Current plan SHA256: `1d2ff7b863c0dc0724b8cb5cc34dadfa18b6dccf8e38c1e29b2015dc67e84076`.

Before reserved tests, archive candidate weights/runtime source and record source commit, checkpoint SHA, plan SHA and hashes of every used runtime source in `final-selection.json`. Test queueing refuses changed files and skips any already-started test, including partial attempts. Source code and evidence reports are committed/pushed; recordings, weights and fitted coefficient arrays stay outside Git.

## Isolation and reproduction

Owned immutable container: `0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`; RTX 4070 UUID `GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`. Retain four CPU cores of quota on 16–23, shares 256, nice 10, 16 GiB RAM/no extra swap. The identity-bound guard yields by stopping only AIWorker if foreign 4070 compute appears. Other workloads, including the 4090 container, are read only. These limits do not imply literally zero shared-host overhead.

Use `tools/deploy_wide_study.py` on the Ubuntu host for an idle-only archived deployment. It verifies ownership, immutable ID, GPU exclusivity and caps; refuses changes after test selection; and restarts only the owned container. Native Python is `/isaac-sim/python.sh`.

Local protocol checks: `python3 -m unittest discover -s tests -p test_wide_protocol.py -v`. Initial preflight and corrected audits are `tools/preflight_wide_study.py` and `tools/audit_wide_clearance.py`. POST commands to `http://192.168.68.60:8212/control`: `wide_repair`, `wide_collect`, `wide_train`, `wide_development`, `wide_test`, only in the appropriate verified idle stage. Do not reload/stop an active recording to advance stages. Final report/video commands and measured outcomes will be added after the actual evaluation.
