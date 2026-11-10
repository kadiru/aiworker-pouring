"""Reviewed development episodes only; no reserved data admission."""
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from scale_data_admission import admit,collection_paths
def entries():
    protected_unchanged();plan=read(ROOT/'plan.json');m=read(ROOT/'candidate.json');p=read(ROOT/'protocol.json')
    ap,rp=collection_paths(ROOT,m);assert read(rp)['accepted'] and read(rp)['audit_sha256']==sha(ap)
    result=list(admit(plan,m,read(ap)))
    for part in ('seed_pilot','seed_remaining','low_train','train','validation'):
        ap=ROOT/(part+'-r0-audit.json');audit=read(ap);review=read(ROOT/(part+'-r0-manual-review.json'))
        assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ap)
        specs=p.get(part)
        if part.startswith('seed_'):
            pilot=set(p['seed_pilot_names']);specs=[s for s in p['seed'] if (s['name'] in pilot)==(part=='seed_pilot')]
        indexed={s['name']:s for s in specs};assert indexed.keys()=={r['name'] for r in audit['records']}
        for r in audit['records']:
            s=indexed[r['name']];assert s['study_split']==r['split'] and r['split'] in ('train','validation')
            assert r['recording_validation']['valid'] and r['label']['accepted']
            if s.get('calibration_motion'):assert r['motion']['passed']
            result.append((s,r,r['split']))
    assert len(result)==207 and len({s['name'] for s,r,split in result})==207
    return result
