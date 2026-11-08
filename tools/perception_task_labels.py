"""Causal measured labels for consumed task images; offline fitting only."""
from pour_perception_geometry import settled_label


def task_level_frames(samples, decisions, stride=2):
    assert len(samples)==len(decisions)
    frames=[]
    for i in range(12,len(samples),stride):
        d=decisions[i]
        if not d.get('visual_grasp_confirmed') or d['cup_xy'] is None:
            continue
        # Level decisions happen during settling before a stop is latched.
        # During the subsequent return the bottle can fully occlude the cup;
        # stable particle height alone does not make that image informative.
        if d['fill_mode']!='settle' or d.get('stop_latched'):
            continue
        # obs[i] precedes action i and matches samples[i-1], not samples[i].
        assert abs(d['t']-samples[i-1]['t'])<.0002
        label=settled_label(samples[:i],seconds=.5)
        if not label['accepted']:
            continue
        # Do not smooth away a late jump in the physical surface.
        if abs(label['target']-samples[i-1]['cup_fill_fraction'])>.01:
            continue
        frames.append(dict(group='obs',index=i,target=label['target'],label=label,
            prototype=f'pause{d["pause_count"]}/level{round(label["target"]/.02)}',
            label_source='measured_causal_half_second_stable_median_before_stop'))
    return frames
