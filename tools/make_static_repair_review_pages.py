"""Concatenate SHA-verified native review images without generated imagery."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=('train',),required=True);a=p.parse_args()
    root=Path('evidence/dynamic-A');stem=a.part+'-r2';audit=json.loads((root/(stem+'-audit.json')).read_text());out=root/(stem+'-review');out.mkdir(exist_ok=True)
    rows=sorted(audit['records'],key=lambda r:r['name']);pages=[]
    for kind,n,field,hashfield in [('final',3,'review_path','review_sha256')]:
        for start in range(0,len(rows),n):
            group=rows[start:start+n];images=[]
            for r in group:
                source=out/Path(r[field]).name;assert sha(source)==r[hashfield];images.append(Image.open(source).convert('RGB'))
            w=max(im.width for im in images);h=sum(im.height for im in images);canvas=Image.new('RGB',(w,h),'white');y=0
            for im in images:canvas.paste(im,(0,y));y+=im.height
            path=out/(kind+'-page-'+str(start//n+1)+'.png');canvas.save(path)
            pages.append(dict(kind=kind,page=str(path),sha256=sha(path),sources=[dict(name=r['name'],sha256=r[hashfield]) for r in group]))
    with (root/(stem+'-review-pages.json')).open('x') as f:json.dump(dict(pages=pages,scope='Exact pixel concatenation of audited native images; no synthetic imagery.'),f,indent=2);f.write('\n')
    print(json.dumps(dict(pages=len(pages),scope='Prepared for actual visual review; not yet reviewed.')))
if __name__=='__main__':main()
