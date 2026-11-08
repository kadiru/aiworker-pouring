"""Explicit perception-training admission; recorded splits are never rewritten."""

def admitted_scenes(plan):
    entries=[(s,s['study_split'],'declared_collection') for s in plan['collect']]
    entries.extend((s,'train','declared_level_supplement') for s in plan.get('level_supplement',[]))
    pilots={s['name']:s for s in plan.get('camera_pilot',[])}
    for name in plan.get('perception_training_pilots',[]):
        s=pilots[name]
        assert s['calibration_only'] and s['study_split'] not in ('train','validation','test')
        entries.append((s,'train','explicit_camera_pilot_reuse'))
    names=[s['name'] for s,_,_ in entries]
    assert len(names)==len(set(names))
    reserved={s['name'] for k in ('development','test','perception_holdout','deferred_size_holdout') for s in plan.get(k,[])}
    assert not reserved.intersection(names)
    assert all(split in ('train','validation') for _,split,_ in entries)
    return entries
