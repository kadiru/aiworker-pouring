"""Autonomous hierarchy: RGB detection, BC pouring, latched terminal primitive.

No object poses, particle inventories, scenario fills or expert paths are read.
Robot joint limits and FK geometry are static hardware configuration. The
terminal trajectory reverses this run's measured joint history, not a test demo.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from pour_learning import Policy,decode_actions,pixels,TemporalEnsemble
from pour_stop_models import LiquidDetector,PourIncrement
from pour_stop_data import crops
from pour_stop_sequence import TerminalPrimitive
from validate_pour_recording import decode


class Actor:
    def __init__(self,checkpoint):
        torch.set_num_threads(4)
        self.saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.base=Policy().cuda().eval();self.base.load_state_dict(self.saved['base_model'])
        self.detector=LiquidDetector().cuda().eval();self.detector.load_state_dict(self.saved['detector'])
        self.pour=PourIncrement().cuda().eval();self.pour.load_state_dict(self.saved['pour'])
        self.hidden=None;self.ensemble=TemporalEnsemble();self.steps=0;self.clamps=0;self.diagnostics=[]
        self.stop_latched=None;self.terminal=None;self.history=[];self.fill_history=[]
        self.start=None;self.filtered_fill=None;self.full_votes=0;self.empty_votes=0;self.lost_votes=0;self.done_votes=0
        self.qlift=None;self.arm_limits=None;self.fk=None;self.lift_rotation=None
        self.pour_phase=0.

    def hardware(self,robot):
        from pxr import UsdPhysics
        from robot_pour import ArmPath,ROOT
        joints=[UsdPhysics.RevoluteJoint.Get(robot.stage,ROOT+'/joints/arm_r_joint'+str(i)) for i in range(1,8)]
        lift=UsdPhysics.PrismaticJoint.Get(robot.stage,ROOT+'/joints/lift_joint')
        self.fk=ArmPath([j.GetLocalPos0Attr().Get() for j in joints],[j.GetAxisAttr().Get() for j in joints],
            [(j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()) for j in joints],lift.GetLocalPos0Attr().Get())
        self.arm_limits=self.fk.limits

    @torch.inference_mode()
    def act(self,observation,robot,t):
        if self.start is None:self.start=t;self.hardware(robot)
        age=t-self.start;q_np=observation['joint_pos'];v_np=observation['joint_vel']
        self.history.append((age,q_np[:7].copy()))
        task=decode(observation['task_encoded'],'jpeg95');wrist=decode(observation['wrist_encoded'],'jpeg95')
        c,s,w=crops(task,wrist)
        vp=self.detector(*[torch.from_numpy(x)[None].cuda() for x in (c,s,w)])
        raw=float(vp['fill'][0])+self.saved['thresholds']['cup_bias'];raw=float(np.clip(raw,0,1))
        self.filtered_fill=raw if self.filtered_fill is None else .65*self.filtered_fill+.35*raw
        self.fill_history.append((t,self.filtered_fill))
        flow=0.
        if len(self.fill_history)>=9:
            recent=np.median([x[1] for x in self.fill_history[-3:]])
            before=np.median([x[1] for x in self.fill_history[-9:-6]])
            flow=float(np.clip((recent-before)/.3*72.8,0,30))
        empty=float(vp['empty_logit'][0].sigmoid());present=float(vp['present_logit'][0].sigmoid())
        profile=self.saved['primitive'];threshold=self.saved['thresholds'];theta=0.
        if self.stop_latched is None and age<profile['lift_mark_s']:
            images=[torch.from_numpy(pixels(x))[None,None].cuda() for x in (task,wrist)]
            q=torch.from_numpy(q_np)[None,None].cuda();v=torch.from_numpy(v_np)[None,None].cuda()
            arm,_,_,_,self.hidden=self.base(*images,q,v,self.hidden)
            chunk=decode_actions(self.base,arm)[0,0].cpu().numpy()
            targets=self.ensemble.apply(chunk,np.zeros(len(chunk),bool))
            phase='Grasping and lifting'
        elif self.stop_latched is None:
            if self.qlift is None:
                self.qlift=q_np[:7].copy();self.lift_rotation=self.fk.fk(self.qlift)[1]
            rotation=self.fk.fk(q_np[:7])[1]
            up=rotation@self.lift_rotation.T@np.array([0.,0.,1.])
            theta=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
            future=self.filtered_fill+flow*(profile['runoff_s']+.15)/72.8
            full=self.filtered_fill>=threshold['full_floor'] and future>=threshold['full_future_target']
            is_empty=empty>=threshold['empty_probability'] and present>=threshold['source_present'] and theta>=95 and flow<3.
            self.full_votes=self.full_votes+1 if full else 0
            self.empty_votes=self.empty_votes+1 if is_empty else 0
            self.lost_votes=self.lost_votes+1 if present<.05 else 0
            reason='cup_near_full' if self.full_votes>=2 else 'bottle_empty' if self.empty_votes>=12 else 'source_tracking_lost' if self.lost_votes>=8 else None
            if reason:
                self.stop_latched=reason;robot.reason=reason;robot.stop_time=t
                self.terminal=TerminalPrimitive(self.history,t,profile)
                targets,phase=self.terminal.act(t)
            else:
                if 'pour_primitive' in self.saved:
                    from pour_stop_trajectory import target
                    movement=self.saved['pour_primitive'];rate=movement['fast_rate_deg_s'] if self.pour_phase<82 else movement['slow_rate_deg_s']
                    if self.filtered_fill>.65:rate*=0. if flow>6. else .55
                    self.pour_phase=min(movement['phase_cap_deg'],self.pour_phase+.05*rate)
                    arm_target=target(movement,self.qlift,self.pour_phase)
                else:
                    images=[torch.from_numpy(pixels(x))[None].cuda() for x in (task,wrist)]
                    features=[enc(x.permute(0,3,1,2).float()/255.) for enc,x in zip((self.base.task,self.base.wrist),images)]
                    q=torch.from_numpy(q_np)[None].cuda();v=torch.from_numpy(v_np)[None].cuda()
                    lift=(torch.from_numpy(self.qlift).cuda()-self.base.q_mean[:7])/self.base.q_std[:7]
                    features+= [(q-self.base.q_mean)/self.base.q_std,v.clamp(-5,5)/2.,lift[None],vp['fill'][:,None],vp['left'][:,None]]
                    delta=(self.pour(torch.cat(features,-1))[0]*.02).cpu().numpy();delta=np.clip(delta,-.02,.02)
                    if self.filtered_fill>.65:delta*=0. if flow>6. else .55
                    arm_target=q_np[:7]+delta
                targets=np.r_[arm_target,np.full(4,np.pi/4)];phase='Pouring until visual stop'
        else:targets,phase=self.terminal.act(t)
        delta=targets-q_np;bounded=np.clip(delta,-np.r_[[.12]*7,[.85]*4],np.r_[[.12]*7,[.85]*4])
        command=q_np+bounded;limited=np.clip(command[:7],self.arm_limits[0],self.arm_limits[1])
        clamped=not np.allclose(delta,bounded) or not np.allclose(command[:7],limited)
        command[:7]=limited;command[7:]=np.clip(command[7:],0,np.pi/4)
        self.clamps+=int(clamped);self.steps+=1;robot.phase='Visual hierarchy: '+phase
        released=np.max(abs(q_np[7:]))<.10 and np.max(abs(v_np))<.08
        terminal_hold=self.terminal is not None and self.terminal.stage==4
        self.done_votes=self.done_votes+1 if terminal_hold and released else 0
        self.diagnostics.append(dict(t=t,mode=phase,stop_latched=self.stop_latched,
            cup_predicted_fraction=self.filtered_fill,source_predicted_ml=float(vp['left'][0])*100,
            empty_probability=empty,source_present_probability=present,visual_flow_ml_s=flow,estimated_tilt_from_joints=theta,
            own_pour_phase_deg=self.pour_phase,
            delta_clamped=bool(clamped),done_probability=float(self.done_votes>=10)))
        return np.degrees(command)

    def complete(self):return self.done_votes>=10
