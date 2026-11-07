"""Native recording integrity and particle accounting, distinct from success.

Runs on the CPU. Reads every closed protocol-18 attempt, including failures;
records partials without imputing their outcomes. No HDF5 attributes are edited.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/src')
from validate_pour_recording import validate,decode,h5py
from pour_finish_study import ROOT,OUTPUTS,PREFIX


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def item(path):
    return dict(relative_path=str(path.relative_to(OUTPUTS)),bytes=path.stat().st_size,sha256=digest(path))


def audit(output,scope="evaluation"):
    files=[];sidecars=[];partials=[];partial_sidecars=[];checks=[];names=set();excluded=[]
    expected={s['name'] for s in json.loads((ROOT/'calibration-plan.json').read_text())['cases']} if scope=='calibration' else None
    for prefix in (("FCAL_A_",) if scope=="calibration" else (PREFIX,)):
        for parameters in sorted(OUTPUTS.glob(f'*/*{prefix}*/parameters.json')):
            spec=json.loads(parameters.read_text())
            if spec.get('protocol')!=18:continue
            if expected is not None and spec['name'] not in expected:
                excluded.append(dict(name=spec['name'],parameters=item(parameters),reason='Superseded unstable calibration initialization; retained, excluded from fit'))
                continue
            if scope=='evaluation' and spec['study_split'] not in ('development','test'):continue
            name=spec['name'];assert name not in names;names.add(name);folder=parameters.parent
            ep=folder/'episode.hdf5';partial=folder/'episode.partial.hdf5'
            if not ep.exists():
                # Never read a partial HDF5 file while its writer is running.
                # Its existence is retained as pending, not a failed checksum.
                partials.append(dict(name=name,path=str(partial),status='incomplete',outcome=None))
                partial_sidecars.append(item(parameters));continue
            assert (folder/'metrics.json').exists() and not partial.exists()
            result=validate(ep);assert result['valid'];metrics=result.pop('metrics')
            samples=json.loads((folder/'samples.json').read_text());quantum=spec['spacing']**3*1e6
            assert len(samples)==result['transitions']
            sidecar_metrics=json.loads((folder/'metrics.json').read_text())
            # HDF5 closes before final wall-time/recording bookkeeping is added.
            # Every physical and controller outcome must nevertheless agree.
            bookkeeping={'wall_seconds','recording'}
            assert {k:v for k,v in metrics.items() if k not in bookkeeping}=={
                k:v for k,v in sidecar_metrics.items() if k not in bookkeeping}
            for row in samples:
                assert abs(sum(row[k] for k in ('bottle_ml','cup_ml','airborne_ml','spilled_ml'))/quantum-row['count'])<1e-5
            assert len({r['count'] for r in samples})==1,'Particle inventory changed'
            assert metrics['spill']['final_outside_ml']==samples[-1]['spilled_ml']+samples[-1]['airborne_ml']
            assert metrics['spill']['ever_table_ml']==max(r.get('ever_table_ml',0.) for r in samples)
            cameras={}
            with h5py.File(ep,'r') as f:
                assert json.loads(f.attrs['spec'])==spec
                assert f.attrs['evidence_kind']==('learned_visual_policy' if spec['policy_evaluation'] else 'scripted_reference')
                assert json.loads(f.attrs['policy_camera_names'])==['task','wrist']
                d=f['data/demo_0'];n=result['transitions']
                for cam in ('task','wrist','overview'):
                    maximum=0.
                    for group,key in (('obs','time'),('next_obs','next_time')):
                        ds=d[group+'/'+cam+'_image'];assert ds.shape==(n,)
                        ts=d[group+'/'+cam+'_time'][:]
                        skew=float(np.max(abs(ts-d[key][:])));maximum=max(maximum,skew)
                        assert skew<1e-6 and (np.diff(ts)>0).all()
                        for i in range(n):
                            frame=decode(ds[i],'jpeg95');assert frame.shape==(384,384,3) and frame.std()>5.
                    for i in range(1,n):
                        assert np.array_equal(d['obs/'+cam+'_image'][i],d['next_obs/'+cam+'_image'][i-1])
                    cameras[cam]=dict(decoded_images=2*n,max_timestamp_skew_s=maximum)
            checks.append(dict(name=name,transitions=result['transitions'],cameras=cameras,particle_inventory_constant=True,
                               all_sample_partitions_accounted=True,visual_review='pending'))
            files.append(dict(name=name,**item(ep),valid=True,transitions=result['transitions'],
                              physical_success=result['physical_success'],max_camera_state_skew_s=result['max_camera_state_skew_s']))
            for filename in ('parameters.json','metrics.json','samples.json','policy-decisions.json'):
                p=folder/filename
                if filename=='policy-decisions.json' and not spec['policy_evaluation']:assert not p.exists();continue
                assert p.exists();sidecars.append(item(p))
            print(json.dumps(dict(name=name,frames=result['transitions'],integrity=True)),flush=True)
    report=dict(files=files,sidecar_files=sidecars,partial_files=[],partial_sidecar_files=partial_sidecars,
                pending_attempts=partials,checks=checks,valid_recordings=len(files),
                native_checks_passed=not partials,transitions=sum(r['transitions'] for r in files),
                scope=scope,excluded_calibrations=excluded,note='Integrity is distinct from task success. No missing outcome is imputed. Manual image review is separate.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='recording-integrity.json');p.add_argument('--scope',choices=('evaluation','calibration','all'),default='evaluation');a=p.parse_args()
    assert Path(a.output).name==a.output
    r=audit(ROOT/a.output,a.scope)
    print(json.dumps(dict(recordings=r['valid_recordings'],pending=len(r['pending_attempts']),transitions=r['transitions'])))
