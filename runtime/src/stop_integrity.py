"""Validate and checksum every completed native S_A recording, failures included."""
import hashlib,json
from pathlib import Path
from validate_pour_recording import validate,export_views
from pour_stop_study import ROOT


def main():
    files=[];partial=[];sidecars=[]
    outputs=Path('/workspace/outputs/fountain')
    for p in sorted(outputs.glob('*/*S_A_*/episode.hdf5')):
        r=validate(p);export_views(p)
        files.append(dict(relative_path=str(p.relative_to(outputs)),bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=r['valid'],transitions=r['transitions'],
            duration_sim_s=r['duration_sim_s'],max_camera_state_skew_s=r['max_camera_state_skew_s'],
            physical_success=r['physical_success']))
        for name in ('metrics.json','parameters.json','samples.json','policy-decisions.json'):
            q=p.parent/name
            sidecars.append(dict(relative_path=str(q.relative_to(outputs)),bytes=q.stat().st_size,
                sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    for p in sorted(outputs.glob('*/*S_A_*/episode.partial.hdf5')):partial.append(str(p))
    report=dict(files=files,sidecar_files=sidecars,valid_recordings=sum(r['valid'] for r in files),partial_aborted_excluded=partial,
        note='Validation establishes record integrity, not task success. Failed native outcomes are retained.')
    (ROOT/'recording-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(valid_recordings=report['valid_recordings'],total_bytes=sum(r['bytes'] for r in files),
        partial_files=len(partial)),indent=2))


if __name__=='__main__':main()
