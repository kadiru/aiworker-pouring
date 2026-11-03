# Transparent-neck grasp confirmation

Implemented follow-up to the completed, frozen `A-guard-20261103-v1` evaluation. The entire earlier six-test block finished and was archived before these controller changes.

The completed `G_A_test_00` physically lifted the bottle with both fingers obstructed and a measured TCP rise of about42mm. Its image-match scores flickered between0.180 and0.289. Although seven of14observations supported the raised neck, the maximum consecutive run was three. The six-consecutive-image rule rejected the valid hold; the retry failed localization and the task ended without pouring or spilling. This remains a failed task, not a zero-spill success.

The repair preserves six consecutive physical confirmations while accepting six visual matches within the last12observations. A failed physical check clears the evidence window. The visual threshold, position/movement checks, both-finger threshold, TCP-rise threshold and all liquid/stop settings remain unchanged. A bounded1.25s verification interval permits noisy image evidence to accumulate. Runtime inputs remain RGB/joints/own history and fixed calibration.

In compact form, enable pouring only when `sum(v[t−11:t]) ≥ 6` and all six latest physical checks pass. Here `v` means a matched, raised neck; a physical check means both fingers obstructed and actual TCP rise>35mm. This is a programmed temporal filter. The cup CNN, empty detector and fitted motor paths retain their trained tensors; no new end-to-end grasp learning is claimed.

The completed `G_A_test_04` also exposes a missed-empty gate: source0, stable visual bottle presence above.99, but empty confidence around.94–.95 below the.97operating point. A slower empty-confirmation branch was calibrated from consumed records, requiring the learned phase cap, low visual flow, continued physical hold and repeated visual evidence. The existing.97/12-observation branch stays unchanged. The new branch uses empty probability>=.90, 24 consecutive observations, at least.25s at the131-degree learned phase cap, measured tilt>=128degrees, both gripper fingers still obstructed, visible-source probability>=.7 and visual flow<1nominalmL/s. These settings were fixed before any H_A tests.

Before a new evaluation, replay the consumed confirmation traces and negative CPU regressions, then run three native regression scenes: rejected grasp `G_A_test_00`, missed empty `G_A_test_04`, and successful full-cup `G_A_test_01`. These scenes are consumed evidence. Select using development only.

Reserve a separate six-case plan under `A-guard-20261103-v2`, prefix `H_A_`, seed20261126, source/cup nominal fills27/5,90/13,37/8,88/14,21/3,98/6, with the existing small position variation. Freeze all controller/model/physics identities before those six tests. Keep the unchanged90%rim-height-or-empty criterion, zero measured escaped particles, full return/release requirement and separate legacy nominal-inventory score. Retain both rounds' results.

On the consumed missed-empty trace, the calibrated branch would trigger at12.305s with source0,1.5s after the empty label, while maintaining90.1%bilateral contact up to the call. No eligible negative-at-cap frames were available; zero observed premature calls therefore does not establish a false-positive rate. Fresh native validation is required. The rejected-grasp trace is recovered by the windowed rule in CPU replay.

Selected checkpoint SHA256: `b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b`. Plan SHA256: `05a0bec06cf0b54b1ce110e61e0fd44617ea2bcb6e4f756d9235da0d6b52ec2a`. Cup/source tensors and motor paths were verified bitwise identical to the trained parent; this follow-up is controller calibration, not another neural fit. The earlier cup-model retraining used the4070. CPU calibration/audits initialize no CUDA context. [Calibration](../evidence/guard-B/vote-calibration.json).

Status: three native regressions running; no new six-case outcome is claimed yet. Thirteen grasp/stop CPU regressions plus the existing19trial-log/GPU-guard checks pass. All lifecycle operations remain confined to the guarded4070 container with existing CPU/RAM limits.
