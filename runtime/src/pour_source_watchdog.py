"""Bounded uncertainty recovery using RGB probabilities and joint evidence only.

Abstention is a failed task, never a substitute success label. This supervisor
does not change the trained empty detector or its success thresholds.
"""
import math


class SourceWatchdog:
    def __init__(self, config):
        self.config = dict(config)
        self.first = self.lost_since = self.uncertain_since = self.cap_since = None
        self.diagnostic = {}

    def update(self, t, *, held, present, empty, tilt, phase, cap, flow_ml_s):
        c = self.config
        if self.first is None:
            self.first = t
        finite = all(math.isfinite(float(v)) for v in (t,present,empty,tilt,phase,cap,flow_ml_s))
        # Early tilted appearances lie outside the old presence detector's
        # reliable range. The separately verified grasp + encoder obstruction
        # carries this short phase; it never licenses an empty success.
        visual_eligible = finite and tilt >= c['presence_min_tilt_deg']
        lost = not held or not finite or (visual_eligible and present < .05)
        ambiguous_empty = finite and held and tilt >= 95 and present < .7 and empty >= .97 and flow_ml_s < 1.
        at_cap = finite and phase >= cap-.25 and tilt >= cap-3.
        for field, active in (('lost_since',lost),('uncertain_since',ambiguous_empty),('cap_since',at_cap)):
            previous = getattr(self, field)
            setattr(self, field, (t if previous is None else previous) if active else None)
        age = lambda start: None if start is None else max(0.,t-start)
        self.diagnostic = dict(visual_presence_eligible=visual_eligible, encoder_hold=bool(held),
            lost_age_s=age(self.lost_since), ambiguous_empty_age_s=age(self.uncertain_since),
            cap_age_s=age(self.cap_since), active_age_s=age(self.first))
        if self.lost_since is not None and t-self.lost_since >= c['lost_seconds']-1e-7:
            return 'source_tracking_lost'
        if self.uncertain_since is not None and t-self.uncertain_since >= c['ambiguous_empty_seconds']-1e-7:
            return 'source_state_unconfirmed'
        if self.cap_since is not None and t-self.cap_since >= c['cap_timeout_seconds']-1e-7:
            return 'pour_progress_unconfirmed'
        if t-self.first >= c['active_timeout_seconds']-1e-7:
            return 'pour_progress_unconfirmed'
        return None


DEFAULT = dict(presence_min_tilt_deg=75.,lost_seconds=.35,
    ambiguous_empty_seconds=1.2,cap_timeout_seconds=8.,active_timeout_seconds=48.)


class TrainedEmptyGate:
    """Consecutive exact-empty wrist predictions corroborated by a low-flow cup."""

    def __init__(self, config):
        self.config = config
        self.votes = 0

    def update(self, *, held, present, tilt, flow_ml_s, cup_ready, upper_level, probability):
        values = (present,tilt,flow_ml_s,upper_level,probability)
        c = self.config
        ready = (held and cup_ready and all(math.isfinite(x) for x in values)
                 and present >= .7 and tilt >= c['tilt']
                 and 0. <= flow_ml_s < c['flow'] and upper_level < .85
                 and probability >= c['probability'])
        self.votes = self.votes+1 if ready else 0
        return self.votes >= c['votes']


class TallBottleEmptyGate:
    """Confirm depletion before the cap using corroborated visual evidence."""

    def __init__(self, use_aligned_scene=False, use_wrist_and_cup=False):
        self.votes = 0
        self.wrist_votes = 0
        self.use_aligned_scene = use_aligned_scene
        self.use_wrist_and_cup = use_wrist_and_cup
        self.scene_view = None

    def update(self, *, held, level_ready, present, wrist_empty, scene_empty, tilt, flow_ml_s,
               aligned_empty=float("nan")):
        finite = all(math.isfinite(x) for x in (present,wrist_empty,tilt,flow_ml_s))
        ready = (finite and held and level_ready and present >= .7 and tilt >= 120.
                 and 0. <= flow_ml_s < 1.)
        self.scene_view = None
        raw_valid = math.isfinite(scene_empty)
        if ready and raw_valid and wrist_empty >= .92 and scene_empty >= .98:
            self.scene_view = "raw"
        elif (ready and raw_valid and self.use_aligned_scene and wrist_empty >= .90 and scene_empty >= .65
              and math.isfinite(aligned_empty) and aligned_empty >= .84):
            self.scene_view = "cup_aligned"
        self.votes = self.votes + 1 if self.scene_view else 0
        self.wrist_votes = self.wrist_votes + 1 if self.use_wrist_and_cup and ready and wrist_empty >= .94 else 0
        if self.votes >= 10:
            return True
        if self.wrist_votes >= 14:
            self.scene_view = "wrist_and_cup"
            return True
        return False
