# First-pour protocol: pour-v1

Proposed engineering acceptance criteria, not ROBOTIS specifications or measured results. Immediate priority is the recovered simulation dry rehearsal in [RECOVERY.md](RECOVERY.md). The physical sequence below is ready for a later hardware session, whose availability remains unknown.

## Scope and labels

One bottle, one cup, fixed marked positions, one arm, fixed base, a small water pour. Start with the bottle already grasped using the operator's established loading procedure. The shot/log must say **“bottle preloaded; fixed cup.”** Picking up a bottle and serving a person are separate milestones. Keep the unused arm parked in an operator-verified configuration.

| Label | Required meaning |
| --- | --- |
| Teleoperated | A person controls movement during the cycle; ordinary continuous operator input is expected |
| Scripted | Deterministic recorded/calibrated commands produce the motion; no trained action policy |
| Learned | A named checkpoint generates task actions from observations; no human action correction within the scored cycle |
| Mixed | Control source changes within the cycle; identify each stage and any checkpoint |
| Simulation | A simulator ran; show its name and disclose rigid attachment/visual-only liquid if used |
| Synthetic | Invented values or software fixtures; never described as simulated robot physics |

An emergency stop or recovery intervention fails a cycle regardless of mode. A person resetting the bottle between trials is outside the scored interval and must be stated. Human corrections during a learned attempt remain recorded against that attempted policy, with `intervention=true`; any final edited footage must be labeled assisted/mixed.

## Setup and measurement

1. Confirm model, payload/gripper suitability, installed control stack, stop behavior, and one active command source with the person responsible for the robot. Verify clearance and stopping with no liquid. Keep the stop control accessible; do not rely on controller self-collision filtering to protect the table or cup.
2. Use a lightweight unbreakable bottle and stable wide-mouth cup over a catch tray away from electronics. Mark bottle-rest and cup footprints/orientation; photograph the fixture with a ruler. Record container IDs/dimensions, table height, camera placement and the locally measured transforms in a versioned setup manifest. No calibrated poses are supplied here.
3. Use water only for the first wet demonstration. Set initial water to **150 ±1 g**, target delivery **50 ±5 g**, and cup capacity comfortably above the entire charge. Confirm the bottle plus contents is within the verified usable payload at the intended posture. These quantities are a proposed experiment, not a payload recommendation for unidentified hardware.
4. Use a scale with displayed resolution no worse than 1 g; record its ID/resolution and repeatability check using a reference mass. If measurement uncertainty cannot resolve these thresholds, revise the protocol before testing and version it. The logger currently implements only `pour-v1`.
5. Before and after each wet cycle, weigh the cup, bottle, and spill collector separately. Use the same containers without changing caps/accessories between paired readings. A removable absorbent liner/tray collects drips, including those after the bottle returns upright. Do not include the cup or bottle in the spill weight. No wiping/refilling before final readings. Document liquid outside containment as an incident even if its mass is unavailable.

## Execution gates

**Dry:** perform three complete cycles with an empty bottle, fixed cup and the intended control mode. Start at the verified pre-grasped upright state; approach, position the mouth above the opening, tilt, return upright, then stop at rest. Check grip, clearances, camera visibility and stop behavior. Log every attempt, including aborts. Any change in grip/fixture requires a new setup ID and dry recheck.

**First wet:** repeat with the measured starting charge, under direct supervision. Record start/end times and continuous video. Begin the scored interval at the first commanded movement from the declared rest; end after upright return and a fixed five-second drip-settling period. Control the pour using the established teleoperator method first. Do not guess a tilt angle/dwell or claim scale feedback if the controller does not read it. A scripted pour must be calibrated locally; a learned pour needs separate policy validation.

**Repeatability:** after a successful first wet trial, freeze fixture, initial water, controller/checkpoint and protocol. Collect a predeclared block of five consecutive attempts. Count failures and aborts, not just saved successes. Require at least **4/5 passing pours** and **no collision, drop, emergency stop, or spill outside the tray in the block**. Report all five delivered masses and spill masses, mode, setup and interventions. This is a demo milestone, not a statistical reliability claim. The logger reports the most recent five within each series; use a new series ID for each predeclared block and retain earlier blocks.

## Per-cycle pass criteria

- Full declared cycle and drip settling completed within **60 seconds**, without a recovery intervention, collision, drop, emergency stop or liquid outside containment.
- Initial water **149–151 g**; delivered mass `cup_after − cup_before` in **45–55 g**; collected spill `spill_after − spill_before` in **0–2 g**.
- Mass accounting: `bottle_before − bottle_after − delivered − spill` has magnitude **≤3 g**; bottle loss is nonnegative and cannot exceed initial water. Negative spill/delivery or missing readings cannot pass.
- Clear evidence of stream landing in the cup and an upright finish; camera/data quality reviewed separately. Numeric logging alone cannot verify the video, grip or task geometry.

The smallest later hardware experiment is **one empty-bottle teleoperated tilt-and-return over the fixed empty cup**, with a continuous clip and command/state trace. Only then proceed through the dry gate to a wet measured pour. Hardware access is not necessary to finish the logger, historical story or simulation recovery.
