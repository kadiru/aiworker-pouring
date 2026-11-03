"""CPU-only plots from saved measurements; never starts Isaac Sim."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--root',default='/workspace/outputs/fountain')
args=p.parse_args()
root=Path(args.root)
data=json.loads((root/'latest.json').read_text())
results=data['results']
out=root/'report';out.mkdir(exist_ok=True)
names={'slosh_baseline','control_viscosity_10000','resolution_1p8','resolution_1p2','resolution_1p0'}
slosh=[r for r in results if r['metrics'].get('slosh') and r['spec']['name'] in names]
fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
fig.suptitle('Isaac Sim fountain: measured behavior, not calibrated waterness',fontsize=17)
coarse=[r for r in results if not r['spec']['name'].startswith(('replay_','render_','validate_','refine_','slosh_','target_','resolution_')) and r['spec'].get('protocol',0)<3]
for r in coarse:
    m=r['metrics'];s=r['spec'];label=s['name']
    axes[0,0].scatter(m['final_rms_m_s']*1000,m['screening_score'],marker='o' if m['eligible'] else 'x',s=45)
    if label in ('baseline','fine','low_friction','control_thick','control_leak'):
        axes[0,0].annotate(label,(m['final_rms_m_s']*1000,m['screening_score']),fontsize=8,xytext=(5,3),textcoords='offset points')
axes[0,0].set(xlabel='Final pool RMS speed (mm/s)',ylabel='Initial screening score / 100',title='Initial scoring can favor a sluggish liquid')
for r in slosh:
    samples=json.loads((Path(r['path'])/'samples.json').read_text())
    rows=[x for x in samples if x['t']>=r['spec']['slosh_time']]
    ts=np.array([x['t']-r['spec']['slosh_time'] for x in rows])
    com=np.array([x['pool_com_x'] for x in rows])*1000
    label=r['spec']['name'].removeprefix('slosh_').replace('resolution_','').replace('1p','1.').replace('control_viscosity_10000','viscosity 10,000 control')
    axes[0,1].plot(ts,com-com[-10:].mean(),label=label,lw=1.3)
axes[0,1].set(xlabel='Seconds after a known 0.08 m/s impulse',ylabel='Pool center-of-mass displacement (mm)',title='Sloshing and decay from physical particles')
axes[0,1].legend(fontsize=9,loc='upper right')
for r in slosh:
    m=r['metrics'];s=m['slosh']
    axes[1,0].scatter(s['frequency_hz'],s['decay_per_s'],s=55,marker='o' if m['eligible'] else 'x')
    axes[1,0].annotate(r['spec']['name'].removeprefix('slosh_').replace('resolution_','').replace('1p','1.').replace('control_viscosity_10000','high viscosity control'),(s['frequency_hz'],s['decay_per_s']),fontsize=8,xytext=(4,3),textcoords='offset points')
axes[1,0].set(xlabel='Fitted oscillation frequency (Hz)',ylabel='Fitted decay rate (1/s)',title='A water-like candidate should sustain a measurable wave')
ranking=sorted([r for r in results if r['metrics'].get('water_screening_score') is not None and r['metrics']['eligible'] and not r['spec']['name'].startswith('replay_')],key=lambda r:r['metrics']['water_screening_score'],reverse=True)[:10]
axes[1,1].barh([r['spec']['name'] for r in ranking][::-1],[r['metrics']['water_screening_score'] for r in ranking][::-1],color='#1695a3')
axes[1,1].set(xlim=(0,100),xlabel='Combined screening score / 100',title='Eligible tested candidates; heuristic weights')
for ax in axes.ravel(): ax.grid(alpha=.15);ax.set_axisbelow(True)
fig.savefig(out/'measurements.png',dpi=160)
fig.savefig(out/'measurements.pdf')
print(json.dumps(dict(trials=len(results),ranked=[dict(name=r['spec']['name'],metrics=r['metrics']) for r in ranking]),indent=2))
