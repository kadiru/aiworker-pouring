"""Retire occluded moving additions; declare separately named static training."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    old=read(ROOT/'protocol.json');assert old['version']==4 and (ROOT/'train-r1-stopped.json').exists()
    probe=read(ROOT/'repair-early-tracker-cpu.json');assert not all(r['necessary_check'] for r in probe['cases'])
    assert not (ROOT/'component-selection.json').exists()
    p=copy.deepcopy(old);wanted=set(old['repair_train_names']);old_extra=[s for s in old['train'] if s['name'] in wanted];new=[]
    for s in old_extra:
        s=copy.deepcopy(s);s['name']=s['name'].replace('motion_train_r1','static_train_r2');s.pop('calibration_motion')
        s.update(coverage_role='static_low_intermediate_repair');new.append(s)
    p['train']=[s for s in old['train'] if s['name'] not in wanted]+new
    p.update(version=5,retired_repair_train_names=old['repair_train_names'],repair_train_names=[s['name'] for s in new],
        previous_protocol_sha256=sha(ROOT/'protocol.json'),repair_rationale='The additive moving r1 batch was stopped after two early CPU tracker rejections, confirmed by six actual native frames showing bottle occlusion. Preserve completed/partial records; none enter fitting. Replace with 36 new static training-only recordings at the same two layouts and .24/.38 cuts, keeping original moving train/validation and all reserved cases/gates unchanged.')
    for k in ('validation','holdout','gates'):assert p[k]==old[k]
    with (ROOT/'protocol-repair-v5-draft.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps(dict(extra_static_training=len(new),draft_sha256=sha(ROOT/'protocol-repair-v5-draft.json'),activated=False)))
if __name__=='__main__':main()
