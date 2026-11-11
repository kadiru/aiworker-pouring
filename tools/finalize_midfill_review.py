"""Bind completed native audit to the three actually inspected review blocks."""
import hashlib,json
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    audit=read(ROOT/'train-r3-audit.json');p=read(ROOT/'protocol.json');assert p['version']==6
    assert audit['complete'] and audit['protocol_sha256']==sha(ROOT/'protocol.json')
    records={r['name']:r for r in audit['records']};assert records.keys()==set(p['midfill_train_names']) and len(records)==36
    assert all(r['recording_validation']['valid'] and r['label']['accepted'] and all(v==0 for v in r['containment'].values()) for r in records.values())
    seen=set();blocks=[]
    for i in (1,2,3):
        manifest=ROOT/f'train-r3-early-review-b{i}.json';m=read(manifest);review=ROOT/f'train-r3-review-b{i}.json';v=read(review)
        assert v['accepted'] and v['manifest_sha256']==sha(manifest) and v['selected_camera_frames']==48
        assert m['protocol_sha256']==sha(ROOT/'protocol.json') and len(m['cases'])==12 and len(m['pages'])==4
        assert all(sha(Path(q['path']))==q['sha256'] for q in m['pages'])
        for c in m['cases']:
            assert c['name'] not in seen and c['source_sha256']==records[c['name']]['sha256'] and c['cameras']==['task','wrist','cup_detail','overview'];seen.add(c['name'])
        blocks.append(dict(batch=i,manifest_sha256=sha(manifest),review_sha256=sha(review)))
    assert seen==records.keys()
    report=dict(accepted=True,audit_sha256=sha(ROOT/'train-r3-audit.json'),protocol_sha256=sha(ROOT/'protocol.json'),episodes=36,selected_camera_frames=144,review_pages=12,blocks=blocks,
        level_range=[min(r['label']['target'] for r in records.values()),max(r['label']['target'] for r in records.values())],all_zero_measured_loss=True,
        scope='All 36 native static records pass recording and stable-label/containment audit; 144 actual final-camera frames inspected on 12 pages. Not continuous playback, autonomous pouring or tracker acceptance.')
    with (ROOT/'train-r3-manual-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
