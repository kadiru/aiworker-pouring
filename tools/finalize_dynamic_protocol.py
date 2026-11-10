"""Bind declared motion cases to all nine reviewed, measured high-fill seeds."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    old=read(ROOT/'protocol.json');new=read(ROOT/'motion-draft.json');assert new['draft_only']
    assert old['seed']==new['seed'] and old['low_train']==new['low_train']
    seeds={};specs={s['name']:s for s in old['seed']}
    for part in ('seed_pilot','seed_remaining'):
        path=ROOT/(part+'-r0-audit.json');a=read(path);review=read(ROOT/(part+'-r0-manual-review.json'))
        assert a['complete'] and review['accepted'] and review['audit_sha256']==sha(path)
        for r in a['records']:
            assert r['recording_validation']['valid'] and r['label']['accepted'] and r['high_fill_coverage']
            assert sha(Path(r['settled_seed_path']))==r['settled_seed_sha256']
            s=specs[r['name']];key=(s['cup_height_scale'],s['cup_diameter_scale']);assert key not in seeds;seeds[key]=r
    assert len(seeds)==9
    for part in ('train','validation','holdout'):
        for s in new[part]:
            if not s['high_seed_binding_pending']:continue
            r=seeds[s['cup_height_scale'],s['cup_diameter_scale']]
            s.update(calibration_particle_seed=r['settled_seed_path'],calibration_particle_seed_sha256=r['settled_seed_sha256'],calibration_initial_height=1.,
                cup_initial_ml=r['settled_seed_particles']*s['spacing']**3*1e6,high_seed_binding_pending=False)
    new['draft_only']=False
    shutil.copy2(ROOT/'protocol.json',ROOT/'protocol-v2.json')
    (ROOT/'protocol.json').write_text(json.dumps(new,indent=2)+'\n')
    m=read(ROOT/'candidate.json');m.update(high_seeds_accepted=True,stage='moving_view_collection')
    (ROOT/'candidate.json').write_text(json.dumps(m,indent=2)+'\n')
    report=dict(high_seeds=9,protocol_sha256=sha(ROOT/'protocol.json'),draft_sha256=sha(ROOT/'motion-draft.json'),helper_sha256=sha(Path(__file__)),autonomy_accepted=False)
    (ROOT/'high-seed-binding.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
