"""Render completed reference camera sheets without modifying recordings.

Run with Isaac's Python inside the existing capped container. These privileged
references are training evidence, not autonomous benchmark outcomes.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/src')
import numpy as np
from PIL import Image, ImageDraw
from validate_pour_recording import decode, h5py
from pour_workspace_study import ROOT


def main():
    files = [p for p in sorted(Path('/workspace/outputs/fountain').glob(
        '*/*W30_A_demo_*/episode.hdf5')) if (p.parent/'metrics.json').exists()]
    assert files
    for page in range((len(files)+3)//4):
        group = files[page*4:(page+1)*4]
        sheet = Image.new('RGB', (1600, 370*len(group)), (20, 25, 32))
        draw = ImageDraw.Draw(sheet)
        for n, path in enumerate(group):
            with h5py.File(path, 'r') as f:
                d = f['data/demo_0']; spec = json.loads(f.attrs['spec'])
                metrics = json.loads(f.attrs['metrics'])
                times = d['next_time'][:]; stop = np.flatnonzero(d['ground_truth/stop_command'][:])
                assert len(stop), spec['name']
                stages = ((0, 'task', 'initial task'),
                    (int(np.argmin(abs(times-3.))), 'wrist', '3s wrist'),
                    (int(np.argmin(abs(times-8.))), 'wrist', '8s or final wrist'),
                    (int(stop[0]), 'task', 'stop task'),
                    (len(times)-1, 'task', 'final task'))
                draw.text((8, n*370+5), f'{spec["name"]} | SCRIPTED REFERENCE | loss={metrics["spill"]["final_outside_ml"]:.4f}', fill='white')
                for col, (ix, cam, label) in enumerate(stages):
                    draw.text((col*320+8, n*370+26), f'{label} ({times[ix]:.2f}s)', fill='white')
                    image = Image.fromarray(decode(d['next_obs/'+cam+'_image'][ix], f.attrs['image_encoding']))
                    image.thumbnail((320, 320))
                    sheet.paste(image, (col*320, n*370+48))
        out = ROOT/f'reference-camera-review-{page+1:02}.jpg'
        sheet.save(out, quality=94)
        print(out, flush=True)


if __name__ == '__main__':
    main()
