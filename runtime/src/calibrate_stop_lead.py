"""One pre-test stop-lead correction, based on consumed native validation."""
import hashlib,json,shutil
from pathlib import Path
import torch
from pour_stop_study import ROOT


def main():
    p=ROOT/'best.pt';before=hashlib.sha256(p.read_bytes()).hexdigest()
    b=torch.load(p,map_location='cpu',weights_only=False)
    assert b['thresholds']['full_floor']==.82,'Lead already calibrated; refusing repeat mutation'
    shutil.copy2(p,ROOT/'before-lead-calibration.pt')
    b['thresholds']['full_floor']=.78
    note=dict(previous_checkpoint_sha256=before,full_floor_before=.82,full_floor_after=.78,
        basis='Consumed V9 validation00: actual surface92.13% at stop, five particles escaped during return near98.27% surface. Existing forecast was blocked by the .82 floor. .78 admits approximately .15s more lead in that saved decision trace.',
        unchanged='all learned weights, future target .91, primary and legacy scoring, empty detector/persistence, motion primitives, physics, frozen scenes',
        calibration_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    b['full_lead_calibration']=note
    torch.save(b,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(p)
    note.update(selected_checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cuda_initialized=torch.cuda.is_initialized())
    (ROOT/'full-lead-calibration.json').write_text(json.dumps(note,indent=2)+'\n')
    print(json.dumps(note,indent=2))


if __name__=='__main__':main()
