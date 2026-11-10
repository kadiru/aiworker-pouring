"""Archive completed failed fits, then activate an additive training protocol."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    old=read(ROOT/'protocol.json');new=read(ROOT/'protocol-repair-v4-draft.json');geo=read(ROOT/'protocol-repair-v4-draft-train-geometry.json')
    assert old['version']==3 and new['version']==4 and new['previous_protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(new[k]==old[k] for k in old if k not in ('version','train')) and new['train'][:len(old['train'])]==old['train']
    assert geo['passed'] and geo['complete'] and geo['protocol_sha256']==sha(ROOT/'protocol-repair-v4-draft.json')
    assert {r['name'] for r in geo['cases']}=={s['name'] for s in new['train']}
    m=read(ROOT/'candidate.json')
    assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    archives={}
    for prefix in ('pair-r5','pair-r6'):
        result=read(ROOT/(prefix+'.json'));cfg=read(ROOT/(prefix+'-protocol.json'))
        assert not result['passed'] and result['protocol_sha256']==sha(ROOT/(prefix+'-protocol.json'))
        assert read(ROOT/(prefix+'-fit/completion.json'))['exit_code']==0
        assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
        folder=ROOT/(prefix+'-frozen-inputs');folder.mkdir()
        files=sorted(set(list(cfg['frozen_files'])+[prefix+'.json',prefix+'-protocol.json',prefix+'-rgb.pt',prefix+'-color.pt',prefix+'-rgb.json',prefix+'-color.json']))
        hashes={}
        for n in files:
            target=folder/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,target);hashes[n]=sha(ROOT/n);assert sha(target)==hashes[n]
        (folder/'archive.json').write_text(json.dumps(hashes,indent=2)+'\n');archives[prefix]=dict(directory=str(folder),files=len(hashes),manifest_sha256=sha(folder/'archive.json'))
    shutil.copy2(ROOT/'protocol.json',ROOT/'protocol-v3-before-level-repair.json')
    shutil.copy2(ROOT/'train-geometry.json',ROOT/'train-geometry-v3-before-level-repair.json')
    (ROOT/'protocol.json').write_bytes((ROOT/'protocol-repair-v4-draft.json').read_bytes())
    (ROOT/'train-geometry.json').write_bytes((ROOT/'protocol-repair-v4-draft-train-geometry.json').read_bytes())
    report=dict(previous_protocol_sha256=sha(ROOT/'protocol-v3-before-level-repair.json'),protocol_sha256=sha(ROOT/'protocol.json'),extra_training=len(new['repair_train_names']),
        archives=archives,validation_unchanged=True,reserved_unchanged=True,gates_unchanged=True,runtime_sources_unchanged=True,tracker_sha256=sha(ROOT/'tracker-t1.pt'),active_checkpoint_sha256=m['checkpoint_sha256'],autonomy_accepted=False)
    with (ROOT/'repair-collection-activation.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
