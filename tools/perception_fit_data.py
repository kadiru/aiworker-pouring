"""Explicit perception-training admission; recorded splits are never rewritten."""

def admitted_scenes(plan):
    entries=[(s,s['study_split'],'declared_collection') for s in plan['collect']]
    entries.extend((s,'train','declared_level_supplement') for s in plan.get('level_supplement',[]))
    declared={n for h in plan.get('perception_history',[]) if not h['passed'] for n in h['consumed_training_names']}
    for s in plan.get('consumed_perception_training',[]):
        assert s['name'] in declared and s['study_split']=='perception_test' and s['calibration_only']
        entries.append((s,'train','explicit_consumed_failed_perception_holdout'))
    pilots={s['name']:s for s in plan.get('camera_pilot',[])}
    for name in plan.get('perception_training_pilots',[]):
        s=pilots[name]
        assert s['calibration_only'] and s['study_split'] not in ('train','validation','test')
        entries.append((s,'train','explicit_camera_pilot_reuse'))
    names=[s['name'] for s,_,_ in entries]
    assert len(names)==len(set(names))
    reserved={s['name'] for k in ('development','test','perception_holdout','deferred_size_holdout','archived_perception_holdout') for s in plan.get(k,[])}
    assert not reserved.intersection(names)
    assert all(split in ('train','validation') for _,split,_ in entries)
    return entries


def admitted_tasks(plan):
    """Consumed development episodes, with fit splits declared before fitting.

    These are development tuning data. Their old task outcomes and recorded
    development split stay intact; reserved test episodes are never admitted.
    """
    declared={name:split for h in plan.get('task_repair_history',[])
              for name,split in h['fit_splits'].items()}
    reserved={s['name'] for k in ('development','test','perception_holdout',
        'deferred_size_holdout','archived_perception_holdout') for s in plan.get(k,[])}
    entries=[]
    for spec in plan.get('consumed_task_development',[]):
        name=spec['name'];split=declared[name]
        assert name not in reserved and spec['study_split']=='development'
        assert split in ('train','validation')
        assert spec['policy_evaluation'] and not spec.get('calibration_only')
        entries.append((spec,split,'explicit_consumed_autonomous_development'))
    names=[s['name'] for s,_,_ in entries]
    assert len(names)==len(set(names)) and set(names)==set(declared)
    assert not set(names).intersection(s['name'] for s,_,_ in admitted_scenes(plan))
    return entries
