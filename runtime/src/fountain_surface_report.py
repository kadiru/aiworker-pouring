"""Contact sheet of matching crops from measured rendering trials."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=Path('/workspace/outputs/fountain')
results=json.loads((root/'latest.json').read_text())['results']
rows=[r for r in results if r['spec']['name'].startswith('render_')]
fig,axes=plt.subplots(2,len(rows),figsize=(3*len(rows),7),squeeze=False,constrained_layout=True)
fig.suptitle('Identical physical preset · surface comparison\nTop: flowing at approximately 2 s. Bottom: settled at approximately 6 s.',fontsize=15)
for column,r in enumerate(rows):
    frames=sorted((Path(r['path'])/'frames').glob('*.jpg'))
    for row,target in enumerate((40,120)):
        path=min(frames,key=lambda p:abs(int(p.stem)-target))
        im=Image.open(path)
        # The same crop in every frame preserves the nozzle and entire receiver.
        axes[row,column].imshow(im.crop((400,80,880,630)))
        axes[row,column].set_axis_off()
    axes[0,column].set_title(r['spec']['name'].removeprefix('render_').replace('_',' '),fontsize=11)
    axes[1,column].set_title(f"Retained {r['metrics']['final_retention']*100:.1f}%",fontsize=10)
(root/'report').mkdir(exist_ok=True)
fig.savefig(root/'report/surface-comparison.png',dpi=150)
fig.savefig(root/'report/surface-comparison.pdf')
print('rendered',len(rows),'comparisons')
