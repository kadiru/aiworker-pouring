"""Actual final-six camera sheet, including failed rollouts; no render synthesis."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from validate_pour_recording import decode,h5py
from pour_stop_study import ROOT


def main():
    r=json.loads((ROOT/'stop-results.json').read_text());assert r['test_status']=='complete'
    rows=sorted(r['groups']['test']['rows'],key=lambda x:x['name'])
    sheet=Image.new('RGB',(1536,444*6),(19,24,29));draw=ImageDraw.Draw(sheet)
    for n,row in enumerate(rows):
        with h5py.File(Path(row['path'])/'episode.hdf5','r') as f:
            d=f['data/demo_0'];stop=np.flatnonzero(d['ground_truth/stop_command'][:]);count=len(d['actions'])
            i=int(stop[0]) if len(stop) else int(np.argmax(d['ground_truth/cup_ml'][:]))
            draw.text((8,n*444+5),f'{row["name"]} | success={row["surface_goal_zero_spill_success"]} | source left={row["source_remaining_ml"]:.3f} | surface={row["cup_surface_height_fraction"]:.3f} | outside={row["final_outside_ml"]:.3f} nominal mL',fill='white')
            for col,(index,camera,label) in enumerate(((i,'task','stop/pour task'),(i,'wrist','stop/pour wrist'),(count-1,'task','final task'),(count-1,'wrist','final wrist'))):
                draw.text((col*384+8,n*444+25),label,fill='white')
                sheet.paste(Image.fromarray(decode(d['next_obs/'+camera+'_image'][index],f.attrs['image_encoding'])),(col*384,n*444+48))
    sheet.save(ROOT/'test-contact-sheet.jpg',quality=94);print(ROOT/'test-contact-sheet.jpg')


if __name__=='__main__':main()
