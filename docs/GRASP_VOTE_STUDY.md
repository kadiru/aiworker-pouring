# Transparent-neck grasp confirmation

Planned follow-up to the frozen `A-guard-20261103-v1` evaluation. No controller change is applied during that six-test block.

The completed `G_A_test_00` physically lifted the bottle with both fingers obstructed and a measured TCP rise of about42mm. Its image-match scores flickered between0.180 and0.289. Although seven of14observations supported the raised neck, the maximum consecutive run was three. The six-consecutive-image rule rejected the valid hold; the retry failed localization and the task ended without pouring or spilling. This remains a failed task, not a zero-spill success.

The proposed repair preserves six consecutive physical confirmations while accepting six visual matches within the last12observations. A failed physical check clears the evidence window. The visual threshold, position/movement checks, both-finger threshold, TCP-rise threshold and all liquid/stop settings remain unchanged. A bounded1.25s verification interval permits noisy image evidence to accumulate. Runtime inputs remain RGB/joints/own history and fixed calibration.

In compact form, enable pouring only when `sum(v[t−11:t]) ≥ 6` and all six latest physical checks pass. Here `v` means a matched, raised neck; a physical check means both fingers obstructed and actual TCP rise>35mm. This is a programmed temporal filter. The cup CNN, empty detector and fitted motor paths retain their trained tensors; no new end-to-end grasp learning is claimed.

The completed `G_A_test_04` also exposes a missed-empty gate: source0, stable visual bottle presence above.99, but empty confidence around.94–.95 below the.97operating point. Calibrate a slower empty-confirmation branch from consumed records, requiring the learned phase cap, low visual flow, continued physical hold and repeated visual evidence. Keep the existing high-confidence branch and all trained tensors; document any selected confidence/time settings before new tests.

Before a new evaluation, replay the consumed confirmation traces and negative CPU regressions, then run three native regression scenes: rejected grasp `G_A_test_00`, missed empty `G_A_test_04`, and successful full-cup `G_A_test_01`. These scenes are consumed evidence. Select using development only.

Reserve a separate six-case plan under `A-guard-20261103-v2`, prefix `H_A_`, seed20261126, source/cup nominal fills27/5,90/13,37/8,88/14,21/3,98/6, with the existing small position variation. Freeze all controller/model/physics identities before those six tests. Keep the unchanged90%rim-height-or-empty criterion, zero measured escaped particles, full return/release requirement and separate legacy nominal-inventory score. Retain both rounds' results.

Status: planned; no follow-up native results yet. Training, testing and all lifecycle operations remain confined to the guarded4070 container with existing CPU/RAM limits.
