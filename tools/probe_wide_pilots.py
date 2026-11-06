"""Optimistic offline framing check on accepted pilots; no fit or test data."""
import os,sys
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/learning/A-wide-20261105-v1/staged','/workspace/src']
from pour_wide_source import probe
from pour_wide_study import ROOT,PARENT
from validate_pour_recording import h5py
from pathlib import Path
import json,torch,copy

torch.set_num_threads(1)
saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
seq=[];localized={}
for p in sorted(Path('/workspace/outputs/fountain').glob('*/*T30_A_demo_*/episode.hdf5')):
    with h5py.File(p) as f:
        s=json.loads(f.attrs['spec']);m=json.loads(f.attrs['metrics'])
        if s.get('protocol')!=17 or not m['grasp_verified'] or not m['zero_spill'] or m['gates']:continue
        if not seq:
            saved['calibration']=dict(K=f['calibration/task/intrinsics'][:].tolist(),
                T=f['data/demo_0/obs/task_world_T_camera'][0].tolist(),table_z=1.04,
                source_reference_calibration=copy.deepcopy(saved['calibration']))
        seq.append(dict(name=s['name'],split=s['study_split'],path=str(p)))
        localized[s['name']]=s['cup_world'][:2]
report=probe(saved,seq,localized,device='cpu')
report['caveat']+=' This pilot framing probe uses labeled cup XY, so it is optimistic; the final probe must use RGB-localized XY.'
(ROOT/'pilot-empty-framing.json').write_text(json.dumps(report,indent=2)+'\n')
for e in report['episodes']:
    positive=[s for s in e['samples'] if s['label_empty']]
    negative=[s for s in e['samples'] if s['source_ml']>=.5]
    print(e['name'], 'positive',len(positive),'detected',sum(s['empty_probability']>=.9 and s['present_probability']>=.7 for s in positive),
        'negative',len(negative),'false_empty',sum(s['empty_probability']>=.9 and s['present_probability']>=.7 for s in negative),flush=True)
