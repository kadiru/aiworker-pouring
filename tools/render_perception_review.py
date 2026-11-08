"""Native still contact sheets for human review, never policy/training crops."""
import argparse,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('collect','level_supplement'),default='collect');args=parser.parse_args()
    audit=read(ROOT/'recording-audit.json');plan=read(ROOT/'plan.json');specs={s['name']:s for s in plan[args.part]}
    records=[r for r in audit['records'] if r['name'] in specs]
    assert len(records)==len(specs) and not audit['started_incomplete']
    records.sort(key=lambda r:(specs[r['name']]['cup_height_scale'],specs[r['name']]['cup_diameter_scale'],r['split'],r['label']['target']))
    folder=ROOT/('collection-review' if args.part=='collect' else 'level-supplement-review');folder.mkdir(exist_ok=False);pages=[]
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),13) if font_path.exists() else ImageFont.load_default()
    for page in range(math.ceil(len(records)/12)):
        selected=records[12*page:12*(page+1)];canvas=Image.new('RGB',(1280,996),'white');draw=ImageDraw.Draw(canvas);entries=[]
        for i,r in enumerate(selected):
            s=specs[r['name']];source=ROOT/'review'/(r['name']+'-cup_detail.jpg');image=Image.open(source).convert('RGB')
            u,v=r['cameras']['cup_detail']['rim_center_uv'];x,y=320*(i%4),332*(i//4)
            # Known pose only positions the human review window; fitting uses
            # image-localized crops and does not read these sheets.
            box=(round(u-120),round(v-65),round(u+120),round(v+175))
            canvas.paste(image.crop(box).resize((320,320),Image.Resampling.BILINEAR),(x,y+12))
            draw.rectangle((x,y,x+320,y+42),fill='white')
            label=f"{r['name']} | {'stable' if r['label']['accepted'] else 'REJECTED'}"
            draw.text((x+5,y+3),label,fill='black',font=font)
            draw.text((x+5,y+21),f"Height {s['cup_height_scale']:.1f}x | diameter {s['cup_diameter_scale']:.1f}x | level {r['label']['target']*100:.1f}%",fill='black',font=font)
            entries.append(dict(name=r['name'],source_sha256=r['sha256'],image_sha256=sha(source),review_crop_box=box))
        output=folder/f'page-{page+1}.png';canvas.save(output)
        pages.append(dict(path=str(output),sha256=sha(output),cases=entries))
    report=dict(pages=pages,audit_sha256=sha(ROOT/'recording-audit.json'),helper_sha256=sha(Path(__file__)),
        scope=f'{len(records)} selected final native cup-detail stills. Known fixture positions are used only for human review windows, not policy inputs or training crops. No generated camera images. Not continuous video review.')
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(pages=len(pages),cases=len(records))))
if __name__=='__main__':main()
