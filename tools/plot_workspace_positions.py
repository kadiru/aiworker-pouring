"""Plot specified positions, not measured outcomes or physical validation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main(plan_path,output):
    p=json.loads(Path(plan_path).read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,6.3))
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.24,wspace=.28)
    for ax,kind,old,title in zip(axes,('cup','bottle'),((8,8),(12,17)),('Cup centers','Initial bottle centers')):
        ax.add_patch(Rectangle((-old[0],-old[1]),old[0]*2,old[1]*2,facecolor='#e8ebef',edgecolor='#76818f',linestyle='--',label='Previous allowed domain'))
        for key,color,marker,label in (
            ('demonstrations','#727b87','^','20 reference scenes'),
            ('development','#c77d17','D','8 development scenes'),
            ('test','#1677ac','o','30 reserved test scenes')):
            xy=[]
            for s in p[key]:
                cx,cy,_=s['cup_world'];sx,sy=s['source_xy']
                xy.append(((cx-.49)*1000,(cy+.15)*1000) if kind=='cup' else ((cx-sy-.49)*1000,(cy+sx+.27)*1000))
            ax.scatter([v[0] for v in xy],[v[1] for v in xy],s=28,c=color,marker=marker,label=label,edgecolors='white',linewidths=.35,zorder=3)
        ax.set_title(title,loc='left',fontweight='bold')
        ax.set_xlabel('World X offset (mm)');ax.set_ylabel('World Y offset (mm)')
        ax.set_aspect('equal');ax.set_xlim(-42,42);ax.set_ylim(-58,58)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Threefold position ranges, with a separate reserved test batch',fontsize=14,fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=2,frameon=False)
    fig.text(.5,.025,'Positions are scene specifications, not outcomes. Origins: cup (0.49, −0.15) m; bottle (0.49, −0.27) m.',ha='center',fontsize=8)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);fig.savefig(output.with_suffix('.pdf'))
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.plan,a.output)
