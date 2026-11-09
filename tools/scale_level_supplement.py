"""Append one intermediate fill at every existing training layout and size."""
import copy
from collections import Counter
def extend(plan,inventory):
    assert plan['version']==5 and len(plan['collect'])==108 and len(plan['holdout'])==9
    sources=[s for s in plan['collect'] if s['study_split']=='train' and s['name'].startswith('SC_A_static_train_')]
    assert len(sources)==36 and len({(s['cup_height_scale'],s['cup_diameter_scale'],s['layout_reference']) for s in sources})==36
    new=[]
    for source in sources:
        s=copy.deepcopy(source);s.update(name=source['name'].replace('SC_A_static_train_','SC_A_level_train_'),calibration_initial_height=.35,level_supplement_of=source['name'])
        s['cup_initial_ml']=inventory[source['name']]['particles']*s['spacing']**3*1e6;new.append(s)
    assert not {s['name'] for s in new}&{s['name'] for s in plan['collect']+plan['holdout']}
    result=copy.deepcopy(plan);result.update(version=6,scope='144 stationary perception episodes: 126 train / 18 unchanged validation; all four training layouts per size gain a seed cut of 0.35. Nine reserved scenes and every accuracy threshold unchanged.')
    result['collect']+=new;result['collection_rounds']['r2']=[s['name'] for s in new]
    count=Counter((s['study_split'],s['cup_height_scale'],s['cup_diameter_scale']) for s in result['collect'])
    for h in (.9,1.,1.1):
        for d in (.9,1.,1.1):assert count['train',h,d]==14 and count['validation',h,d]==2
    assert result['holdout']==plan['holdout'] and result['component_gates']==plan['component_gates']
    assert result['collect'][:108]==plan['collect']
    return result
