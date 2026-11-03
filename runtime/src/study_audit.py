"""CPU recording checks and camera contact sheets; does not approve visual review."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from validate_pour_recording import validate


def audit():
    root=Path('/workspace/learning/A-20261102-v1')
    root.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((root/'frozen-plan.json').read_text())
    names={s['name'] for part in ('train','validation') for s in frozen[part]}
    latest={}
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*A_*/episode.hdf5')):
        params=json.loads((path.parent/'parameters.json').read_text())
        if params['name'] in names: latest[params['name']]=path
    reports=[]
    for name,path in sorted(latest.items()):
        report=path.parent/'validation.json'
        if report.exists(): result=json.loads(report.read_text())
        else:
            result=validate(path)
            report.write_text(json.dumps(result,indent=2)+'\n')
        reports.append(dict(name=name,**result))
    (root/'collection-audit.json').write_text(json.dumps(reports,indent=2)+'\n')
    for start in range(0,len(reports),4):
        subset=reports[start:start+4]
        for camera in ('task','wrist'):
            image=Image.new('RGB',(256*4,280*len(subset)),(32,32,32))
            draw=ImageDraw.Draw(image)
            for row,r in enumerate(subset):
                folder=Path(r['path']).parent
                labels=('initial','approach','grasp','final') if r.get('task_scope')=='approach_lift_prefix' else ('initial','grasp','stop','final')
                for col,label in enumerate(labels):
                    src=folder/(label+'-'+camera+'.png')
                    if src.exists(): image.paste(Image.open(src).resize((256,256)),(col*256,row*280+24))
                draw.text((8,row*280+5),r['name']+' | '+str(r['physical_success']),fill='white')
            image.save(root/f'review-{start//4:02}-{camera}.jpg',quality=94)
    print(json.dumps(dict(completed=len(reports),physical_pass=sum(r['physical_success'] for r in reports),
          rows=[dict(name=r['name'],transitions=r['transitions'],eligible=r['physical_success'],
                     spill=r['metrics'].get('spill'),zero_spill=r['metrics'].get('zero_spill'),
                     gates=r['metrics']['gates']) for r in reports]),indent=2))


if __name__=='__main__':
    audit()
