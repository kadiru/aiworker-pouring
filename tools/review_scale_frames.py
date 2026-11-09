"""Fetch SHA-verified final native camera composites and make review sheets.

Never declares that a person or model inspected the images. Each sheet must be
opened before a separately bound manual-review record can be accepted.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'evidence/scale-A'
NATIVE='/home/kadir/aiworker-pouring-4070/learning/A-scale-20261109-v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('collect','holdout'),required=True);a=parser.parse_args()
    audit_path=ROOT/f'{a.part}-r0-audit.json';audit=read(audit_path);plan_path=ROOT/'plan-collection-r0.json';plan=read(plan_path)
    assert audit['complete'] and audit['plan_sha256']==sha(plan_path)
    if a.part=='holdout':
        selected=read(ROOT/'component-selection.json')
        assert selected['all_size_component_gates_passed'] and selected['plan_sha256']==audit['plan_sha256']
    specs={s['name']:s for s in plan[a.part]};assert {r['name'] for r in audit['records']}==set(specs)
    output=ROOT/f'{a.part}-r0-camera-review';output.mkdir()
    subprocess.run(['scp','-r',f'kfuUbuntu:{NATIVE}/{a.part}-r0-review',str(output)],check=True)
    rows=sorted(audit['records'],key=lambda r:(specs[r['name']]['cup_height_scale'],specs[r['name']]['cup_diameter_scale'],r['name']))
    reviewed={};sheets=[]
    for start in range(0,len(rows),6):
        canvas=Image.new('RGB',(1536,1164),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[start:start+6]):
            original=output/f'{a.part}-r0-review'/f'{r["name"]}.jpg'
            assert sha(original)==r['review_sha256'];image=Image.open(original);assert image.size==(1536,408)
            s=specs[r['name']];x=(j%3)*512;y=(j//3)*582
            canvas.paste(image.crop((768,24,1152,408)).resize((512,512)),(x,y+48))
            draw.text((x+8,y+8),r['name'],fill='black')
            draw.text((x+8,y+26),f'H {s["cup_height_scale"]:.2f}  D {s["cup_diameter_scale"]:.2f} | measured {100*r["label"]["target"]:.2f}% | stable {r["label"]["accepted"]}',fill='black')
            reviewed[r['name']]=dict(native_composite_sha256=r['review_sha256'],source_recording_sha256=r['sha256'])
        path=output/f'sheet-{start//6:02d}.png';canvas.save(path);sheets.append(dict(path=str(path),sha256=sha(path),cases=[r['name'] for r in rows[start:start+6]]))
    report=dict(audit_sha256=sha(audit_path),helper_sha256=sha(Path(__file__)),manual_review='pending',
        cases=reviewed,sheets=sheets,scope='Native final cup-detail panels cropped from SHA-verified four-camera composites. All originals retained. Sheets must actually be inspected; this helper does not accept the data.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),sheets=len(sheets),manual_review='pending',output=str(output))))


if __name__=='__main__':main()
