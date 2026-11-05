"""Combine the original twelve and two additional frozen tests, retaining all."""
import json
from pathlib import Path
from backup_workspace_recordings import digest

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/layout-variety-A';DATA=ROOT/'data/material-A-layout-variety'
def read(name):return json.loads((E/name).read_text())
def write(name,value):(E/name).write_text(json.dumps(value,indent=2)+'\n')

a=read('original12-final-benchmark.json');b=read('extra2-final-benchmark.json')
assert a['status']==b['status']=='complete' and len(a['cases'])==12 and len(b['cases'])==2
assert a['selection']['selected_checkpoint_sha256']==b['selection']['selected_checkpoint_sha256']
sa=a['selection']['runtime_sources'];sb=b['selection']['runtime_sources']
assert {n for n in sa if sa[n]!=sb[n]}=={'run_fountain.py'}
assert set(sb)-set(sa)=={'pour_layout_extra.py'}
rows=sorted(a['cases']+b['cases'],key=lambda r:r['layout_index'])
assert [r['layout_index'] for r in rows]==list(range(14))
assert len({(tuple(r['cup_world']),tuple(r['source_xy'])) for r in rows})==14
counts={k:a['counts'][k]+b['counts'][k] for k in a['counts']}
assert counts['unflagged_starting_layouts']>=10
report=dict(status='complete',counts=counts,cases=rows,
    selections=dict(original_twelve=a['selection'],additional_two=b['selection']),
    primary_definition=a['primary_definition'],
    scope='Fourteen deliberately chosen layouts with five exact swaps; eleven starts without the initial geometry/contact flags. Same learned actor and checkpoint, no retraining. All original twelve plus both supplements retained; not iid success-rate evidence.',
    setup_note='Post-hoc full-hand geometry flags original02/03/08. Native bottle/finger contact is recorded separately; cup contact in03 was not instrumented. Two full-hand-screened additional layouts ensure at least ten unflagged starts without deleting the initial failures.',
    note='PASS means the strict full task. No-pour zero-loss aborts are failures. Material A;400Hz physics and20Hz observations; nominal mL is particle-count based, not calibrated physical volume.')
ia=read('original12-recording-integrity.json');ib=read('extra2-recording-integrity.json')
integrity=dict(files=ia['files']+ib['files'],sidecar_files=ia['sidecar_files']+ib['sidecar_files'],
    checks=ia['checks']+ib['checks'],recordings=14,transitions=ia['transitions']+ib['transitions'],
    partial_files=[],native_checks_passed=ia['native_checks_passed'] and ib['native_checks_passed'])
assert len(integrity['files'])==14
items=integrity['files']+integrity['sidecar_files'];assert len({i['relative_path'] for i in items})==len(items)
for item in items:
    p=DATA/item['relative_path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],p
write('final-benchmark.json',report);write('recording-integrity.json',integrity)
write('backup-verification.json',dict(all_native_hashes_and_sizes_match=True,recordings=14,
    sidecars=len(integrity['sidecar_files']),bytes=sum(i['bytes'] for i in items),partials=0,
    integrity_manifest_sha256=digest(E/'recording-integrity.json')))
print(json.dumps(counts,indent=2))
