"""Latched terminal motor primitive, using only the robot's own measured history."""
import numpy as np


def path_at(path,progress):
    path=np.asarray(path)
    if len(path)<2:return path[0].copy()
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    unique=np.r_[True,np.diff(arc)>1e-8];arc=arc[unique];path=path[unique]
    if len(path)==1:return path[0].copy()
    return np.array([np.interp(np.clip(progress,0,1)*arc[-1],arc,path[:,j]) for j in range(7)])


class TerminalPrimitive:
    def __init__(self,history,start,profile):
        self.start=start;self.profile=profile
        times=np.array([x[0] for x in history]);q=np.array([x[1] for x in history])
        grasp=int(np.argmin(abs(times-profile['grasp_mark_s'])))
        lift=int(np.argmin(abs(times-profile['lift_mark_s'])))
        self.return_path=q[lift:][::-1].copy();self.place_path=q[grasp:lift+1][::-1].copy()
        self.retreat_path=q[:grasp+1][::-1].copy();self.stage=0

    def act(self,t):
        p=self.profile;elapsed=max(0.,t-self.start);ends=np.cumsum([p['return_s'],p['place_s'],p['release_s'],p['retreat_s']])
        stage=int(np.searchsorted(ends,elapsed,side='right'));self.stage=max(self.stage,stage)
        grid=np.linspace(0,1,len(p['warp']));closed=np.pi/4
        if stage==0:
            u=np.interp(elapsed/p['return_s'],grid,p['warp']);q=path_at(self.return_path,u);phase='Returning upright'
        elif stage==1:
            u=(elapsed-ends[0])/p['place_s'];q=path_at(self.place_path,u*u*(3-2*u));phase='Setting bottle down'
        elif stage==2:
            u=(elapsed-ends[1])/p['release_s'];q=self.place_path[-1].copy();closed*=np.interp(u,np.linspace(0,1,len(p['grip'])),p['grip']);phase='Releasing bottle'
        elif stage==3:
            u=(elapsed-ends[2])/p['retreat_s'];q=path_at(self.retreat_path,u*u*(3-2*u));closed=0.;phase='Retreating'
        else:q=self.retreat_path[-1].copy();closed=0.;phase='Holding completed pose'
        return np.r_[q,np.full(4,closed)],phase
