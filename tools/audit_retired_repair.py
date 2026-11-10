"""Preserve every started, aborted and unstarted case from the stopped batch."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import validate
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    assert not (ROOT/'retired-moving-r1-audit.json').exists()
    p=read(ROOT/'protocol-repair-v4-draft.json');names=set(p['repair_train_names']);found={}
    for path in Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_train_r1_*/parameters.json'):
        s=read(path);assert s['name'] not in found;found[s['name']]=path.parent
    rows=[]
    for name in sorted(names):
        row=dict(name=name,status='not_started',admitted_for_training=False)
        if name in found:
            d=found[name];h=d/'episode.hdf5';row.update(directory=str(d),files={v.name:dict(bytes=v.stat().st_size,sha256=sha(v)) for v in d.iterdir() if v.is_file()})
            if h.exists():
                row.update(status='complete_not_admitted',recording_validation=validate(h));samples=read(d/'samples.json')
                row['containment']=dict(final_outside_ml=samples[-1]['spilled_ml'],ever_table_ml=max(v['ever_table_ml'] for v in samples))
            else:row['status']='aborted_not_admitted'
        rows.append(row)
    report=dict(cases=rows,counts={s:sum(r['status']==s for r in rows) for s in ('not_started','complete_not_admitted','aborted_not_admitted')},
        protocol_sha256=sha(ROOT/'protocol-repair-v4-draft.json'),stop_sha256=sha(ROOT/'train-r1-stopped.json'),early_tracker_sha256=sha(ROOT/'repair-early-tracker-cpu.json'),preview_sha256=sha(ROOT/'repair-r1-rejection-preview.jpg'),
        scope='Early stopped training-only motion extension. All complete and partial artifacts retained, no retries or training admission; original moving validation and reserved cases untouched.')
    (ROOT/'retired-moving-r1-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['counts']))
if __name__=='__main__':main()
