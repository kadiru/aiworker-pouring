"""Audit one closed development recording for an early preview while peers run.

Reads only the named, already finalized episode. This is not the complete-round
audit required for candidate selection and does not alter the simulator.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
CID = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'

NATIVE = r'''
import json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/src')
from audit_finish_study import item
from validate_pour_recording import validate,decode,h5py,np
name=sys.argv[1]
root=Path('/workspace/outputs/fountain')
matches=[p for p in root.glob('*/*'+name+'*/parameters.json') if json.loads(p.read_text())['name']==name]
assert len(matches)==1
folder=matches[0].parent;ep=folder/'episode.hdf5'
assert ep.exists() and not (folder/'episode.partial.hdf5').exists(), 'Recording is still active'
result=validate(ep);assert result['valid']
spec=json.loads(matches[0].read_text());assert spec['protocol']==18 and spec['study_split']=='development'
samples=json.loads((folder/'samples.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
assert len(samples)==result['transitions']
assert {k:v for k,v in result['metrics'].items() if k not in ('wall_seconds','recording')}=={k:v for k,v in metrics.items() if k not in ('wall_seconds','recording')}
quantum=spec['spacing']**3*1e6
assert len({r['count'] for r in samples})==1
for r in samples:assert abs(sum(r[k] for k in ('bottle_ml','cup_ml','airborne_ml','spilled_ml'))/quantum-r['count'])<1e-5
with h5py.File(ep) as f:
 assert json.loads(f.attrs['spec'])==spec
 d=f['data/demo_0'];n=result['transitions']
 for camera in ('task','wrist','overview'):
  for group,key in (('obs','time'),('next_obs','next_time')):
   ds=d[group+'/'+camera+'_image'];assert ds.shape==(n,)
   ts=d[group+'/'+camera+'_time'][:]
   assert np.max(abs(ts-d[key][:]))<1e-6 and (np.diff(ts)>0).all()
   for i in range(n):
    im=decode(ds[i],'jpeg95');assert im.shape==(384,384,3) and im.std()>5
  for i in range(1,n):assert np.array_equal(d['obs/'+camera+'_image'][i],d['next_obs/'+camera+'_image'][i-1])
print(json.dumps(dict(files=[dict(name=name,**item(ep),valid=True,transitions=n)],
 sidecar_files=[item(folder/f) for f in ('parameters.json','metrics.json','samples.json','policy-decisions.json')],
 pending_attempts=[],native_checks_passed=True,valid_recordings=1,
 scope='Only the named closed development recording. Other active or queued cases are not assessed. Not the complete-round selection audit.')))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    assert re.fullmatch(r'FN30_A_dev_0[0-7]_r[0-9]+',a.name)
    output=PROJECT/'evidence/finish-A'/(a.name+'-preview-integrity.json')
    assert not output.exists()
    command=shlex.join(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',NATIVE,a.name])
    raw=subprocess.check_output(['ssh','kfuUbuntu',command],text=True,timeout=180)
    report=json.loads(raw.strip().splitlines()[-1])
    assert report['files'][0]['name']==a.name and report['native_checks_passed']
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
