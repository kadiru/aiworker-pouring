"""Compare recorded physical hinge response with a stationary control, CPU only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from fountain_hinge import touchdown

root=Path('/workspace/outputs/fountain')
results=json.loads((root/'latest.json').read_text())['results']
moving=next(r for r in reversed(results) if r['spec'].get('hinge_angle_deg',0)>0)
control=next(r for r in reversed(results) if r['spec']['name']=='hinge_stationary_control')
out=root/'hinge-report';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
fig.suptitle('Hinged receiver: measured water motion around two fixed blocks',fontsize=16)
for result,label,color in [(control,'Stationary receiver','#9c9fa3'),(moving,'22° edge lift and return','#008fa3')]:
    rows=[r for r in json.loads((Path(result['path'])/'samples.json').read_text()) if r['t']>=3.]
    t=np.array([r['t'] for r in rows])
    axes[0,0].plot(t,[r['hinge_angle_deg'] for r in rows],label=label,color=color)
    axes[0,1].plot(t,[r['pool_rms_speed'] for r in rows],label=label,color=color)
    axes[1,0].plot(t,[r['pool_com_x']*1000 for r in rows],label=label,color=color)
    axes[1,1].plot(t,[r['surface_std_mm'] for r in rows],label=label,color=color)
for ax in axes.ravel():
    ax.axvspan(moving['spec']['hinge_start'],touchdown(moving['spec']),color='#eeb450',alpha=.16)
    ax.axvline(touchdown(moving['spec']),color='#c87b13',ls=':',lw=1)
    ax.set(xlabel='Simulation time (s)',xlim=(3,10));ax.grid(alpha=.15)
axes[0,0].set(title='Container motion',ylabel='Tilt angle (degrees)');axes[0,0].legend()
axes[0,1].set(title='Fluid motion relative to the container',ylabel='RMS speed (m/s)')
axes[1,0].set(title='Side-to-side redistribution',ylabel='Pool center x (mm)')
axes[1,1].set(title='Surface response',ylabel='Spatial height variation (mm)')
fig.savefig(out/'response.png',dpi=160);fig.savefig(out/'response.pdf')
fig2,panels=plt.subplots(1,4,figsize=(16,4.5),constrained_layout=True)
for ax,t,title in zip(panels,[3.6,4.6,5.3,6.0],['Filled · ≈3.6 s','One edge lifted · ≈4.6 s','After return · ≈5.3 s','Waves around blocks · ≈6.0 s']):
    index=int(round(t/.1))*2
    im=Image.open(Path(moving['path'])/'frames'/f'{index:04d}.jpg')
    ax.imshow(im);ax.set_title(title,fontsize=11);ax.axis('off')
fig2.savefig(out/'sequence.png',dpi=160)
summary=dict(moving=moving,control=control,
    post_return_peak_speed_ratio=moving['metrics']['hinge']['peak_post_return_rms_m_s']/max(control['metrics']['hinge']['peak_post_return_rms_m_s'],1e-9),
    caveats=['Kinematic hinge trajectory; no gravity drop or measured table impact force.',
             'Fluid velocities are not injected. Response follows actual moving-collider simulation.',
             'Block proximity and COM reversals are geometric proxies, not contact impulses.',
             'Surface images are asynchronous and can lag a rendered frame. Water realism remains provisional.'])
(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
