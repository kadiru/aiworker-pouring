"""All 279 reviewed development episodes, with unchanged validation/reserved sets."""
from pathlib import Path
from perception_repair_data import entries as previous_entries
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def entries():
    protected_unchanged()
    result = previous_entries()
    protocol = read(ROOT/'protocol.json')
    audit = read(ROOT/'train-r3-audit.json')
    review = read(ROOT/'train-r3-manual-review.json')
    assert audit['complete'] and review['accepted'] and review['audit_sha256'] == sha(ROOT/'train-r3-audit.json')
    specs = {s['name']: s for s in protocol['train'] if s['name'] in protocol['midfill_train_names']}
    assert len(specs) == 36 and specs.keys() == {r['name'] for r in audit['records']}
    for r in audit['records']:
        s = specs[r['name']]
        assert s['study_split'] == r['split'] == 'train'
        assert r['recording_validation']['valid'] and r['label']['accepted']
        assert all(v == 0 for v in r['containment'].values())
        assert sha(Path(r['path'])) == r['sha256']
        result.append((s, r, 'train'))
    assert len(result) == len({s['name'] for s, r, split in result}) == 279
    assert len({r['sha256'] for s, r, split in result}) == 279
    assert sum(split == 'train' for s, r, split in result) == 243
    assert sum(split == 'validation' for s, r, split in result) == 36
    reserved = {s['name'] for s in protocol['holdout']+read(ROOT/'plan.json')['holdout']}
    assert not reserved.intersection(s['name'] for s, r, split in result)
    return result
