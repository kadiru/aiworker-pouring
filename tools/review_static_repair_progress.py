"""Prepare native final-camera pages for one closed twelve-case block."""
import argparse,hashlib,json,sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(1,2,3),required=True);a=parser.parse_args()
    p=read(ROOT/'protocol.json');assert p['version']==5
    wanted=p['repair_train_names'][(a.batch-1)*12:a.batch*12];closed={c['name']:c for c in read(ROOT/'train-r2/progress.json')['cases']}
    assert len(wanted)==12 and set(wanted)<=closed.keys()
    folder=ROOT/'train-r2-early-review';folder.mkdir(exist_ok=True);rows=[];images=[]
    for name in wanted:
        path=Path(closed[name]['path'])/'episode.hdf5';assert path.exists()
        canvas=Image.new('RGB',(1536,408),(245,245,245));draw=ImageDraw.Draw(canvas)
        with h5py.File(path) as f:
            d=f['data/demo_0'];index=len(d['next_time'])-1
            for i,camera in enumerate(('task','wrist','cup_detail','overview')):
                im=decode(d['next_obs/'+camera+'_image'][index],'jpeg95');canvas.paste(Image.fromarray(im).resize((384,384)),(i*384,24));draw.text((i*384+6,6),name+' | '+camera,fill=(20,20,20))
        images.append(canvas);rows.append(dict(name=name,path=str(path),source_sha256=sha(path),index=index,cameras=['task','wrist','cup_detail','overview']))
    pages=[]
    for j in range(4):
        page=Image.new('RGB',(1536,1224),'white')
        for k in range(3):page.paste(images[j*3+k],(0,k*408))
        path=folder/f'batch-{a.batch}-page-{j+1}.jpg';page.save(path,quality=95)
        pages.append(dict(path=str(path),sha256=sha(path),names=wanted[j*3:j*3+3]))
    report=dict(batch=a.batch,cases=rows,pages=pages,protocol_sha256=sha(ROOT/'protocol.json'),helper_sha256=sha(Path(__file__)),manual_review='pending',scope='Actual final frames from twelve closed native static records, four cameras each. No generated images, continuous playback, or training acceptance.')
    with (ROOT/f'train-r2-early-review-b{a.batch}.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared_cases=len(rows),pages=len(pages))))
if __name__=='__main__':main()
