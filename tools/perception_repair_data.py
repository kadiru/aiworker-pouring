"""243 reviewed development episodes; immutable whole-episode split boundaries."""
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from scale_data_admission import admit,collection_paths

def entries():
    protected_unchanged()
    plan=read(ROOT/'plan.json');candidate=read(ROOT/'candidate.json')
    protocol=read(ROOT/'protocol.json');original=read(ROOT/'protocol-v3-before-level-repair.json')
    assert all(protocol[k]==original[k] for k in ('validation','holdout','gates'))
    ap,rp=collection_paths(ROOT,candidate)
    assert read(rp)['accepted'] and read(rp)['audit_sha256']==sha(ap)
    result=list(admit(plan,candidate,read(ap)))
    for part in ('seed_pilot','seed_remaining','low_train','train','validation','train-r2'):
        stem=part if part=='train-r2' else part+'-r0'
        ap=ROOT/(stem+'-audit.json');audit=read(ap);review=read(ROOT/(stem+'-manual-review.json'))
        assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ap)
        specs=original.get(part)
        if part.startswith('seed_'):
            pilot=set(original['seed_pilot_names'])
            specs=[s for s in original['seed'] if (s['name'] in pilot)==(part=='seed_pilot')]
        if part=='train-r2':
            specs=[s for s in protocol['train'] if s['name'] in protocol['repair_train_names']]
        indexed={s['name']:s for s in specs}
        assert indexed.keys()=={r['name'] for r in audit['records']}
        for r in audit['records']:
            s=indexed[r['name']]
            assert s['study_split']==r['split'] and r['split'] in ('train','validation')
            assert r['recording_validation']['valid'] and r['label']['accepted']
            if s.get('calibration_motion'):assert r['motion']['passed']
            result.append((s,r,r['split']))
    assert len(result)==243 and len({s['name'] for s,r,split in result})==243
    reserved={s['name'] for s in protocol['holdout']+plan['holdout']}
    assert not reserved.intersection(s['name'] for s,r,split in result)
    assert len({r['sha256'] for s,r,split in result})==243
    assert sum(split=='train' for s,r,split in result)==207
    assert sum(split=='validation' for s,r,split in result)==36
    return result
