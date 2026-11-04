"""Visual servo grasp/retry and settled-level pouring, with no live object state.

Only RGB, measured joints, own history, and fixed sensor/hardware calibration
enter control. Simulator contacts, object poses and particle counts are scoring
information and are deliberately never read here.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from pour_stop_actor import Actor as Hardware
from pour_stop_models import LiquidDetector
from pour_stop_data import crops
from pour_guard_models import CupEstimator
from pour_guard_vision import locate,plane_point,project,cup_crop,surface_features,bottle_base_from_neck
from pour_guard_trajectory import target
from pour_stop_sequence import TerminalPrimitive,path_at
from validate_pour_recording import decode


def ease(x):
    u=np.clip(x,0,1);return u*u*(3-2*u)


def monotone_return_path(envelope,phase,current,lift):
    # Exclude earlier top-up peaks: reversing every pause/re-tip would pour again.
    path=[q for a,q in envelope if a<phase-1.]
    return np.vstack([current]+path[::-1]+[lift])


def topup_goal(settled,target,increment=.015):
    if settled>=target:raise ValueError('Never add liquid when the settled estimate is already full')
    return min(target+.005,settled+increment)


def trim_expired(start,t,seconds):
    # A biased or saturated visual estimate must never permit an endless pour.
    return start is not None and t-start>=seconds-1e-7


def held_gripper(q,gap):
    driven=np.asarray(q)[[7,9]]
    return bool(np.min(driven)>.35 and np.min(np.pi/4-driven)>gap)


def slow_empty_evidence(threshold,phase,cap,tilt,cap_age,held,present,empty,flow_ml_s):
    return bool(cap_age is not None and cap_age>=threshold['empty_cap_dwell_seconds'] and
        phase>=cap-.25 and tilt>=cap-3. and held and present>=.7 and
        empty>=threshold['slow_empty_probability'] and flow_ml_s<threshold['slow_empty_max_flow_ml_s'])


class Actor:
    def __init__(self,checkpoint):
        torch.set_num_threads(4);self.saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.detector=LiquidDetector().cuda().eval();self.detector.load_state_dict(self.saved['source_detector'])
        self.cup_model=CupEstimator().cuda().eval();self.cup_model.load_state_dict(self.saved['cup_model'])
        self.cal=self.saved['calibration'];self.profile=dict(self.saved['primitive']);self.threshold=self.saved['thresholds']
        self.steps=0;self.clamps=0;self.diagnostics=[];self.history=[];self.start=None;self.fk=None;self.arm_limits=None
        self.stop_latched=None;self.terminal=None;self.done_votes=0;self.abort_time=None
        self.grasp_stage='initialize';self.stage_start=None;self.attempt=0;self.confirm_votes=0;self.grasp_confirmed=False
        self.qlift=None;self.lift_rotation=None;self.pour_phase=0.;self.peak_phase=0.;self.fill_mode='coarse'
        self.cup_xy=None;self.source_xy=None;self.filtered=None;self.level_history=[];self.empty_votes=0;self.lost_votes=0
        self.full_votes=0;self.pause_count=0;self.pause_start=None;self.brake_phase=None
        self.trim_goal=None;self.trim_min_phase=None;self.trim_start=None;self.trim_entered=None
        self.outbound=[]
        self.grasp_check=None
        self.visual_votes=[];self.physical_votes=0;self.cap_start=None;self.slow_empty_votes=0
        self.settled_history=[]
        self.trim_cap=95.

    def initialize_grasp(self,task,q,t,age):
        from robot_pour import WRIST
        self.attempt+=1;self.confirm_votes=0;self.visual_votes=[];self.physical_votes=0
        neck,self.source_score=locate(task,self.saved['source_bank']);rim,self.cup_score=locate(task,self.saved['cup_bank'])
        if self.source_score<self.threshold['source_localization_score'] or self.cup_score<self.threshold['cup_localization_score']:
            self.abort_time=t;self.stop_latched='localization_unconfirmed';self.home=q[:7].copy();return
        base=bottle_base_from_neck(neck,self.cal);self.source_xy=base[:2]
        self.cup_xy=plane_point(rim,self.cal['table_z']+.038,self.cal)[:2]
        self.initial_neck=neck;self.expected_lift_neck=project(base+[0,0,.135+.044],self.cal)
        self.home=q[:7].copy()
        # The same known hardware geometry/IK branch as the demonstration planner.
        self.fk.q=np.array([-.25,-.45,-.15,-1.4,0.,.25,0.])
        self.grasp=np.radians(self.fk.solve_pose(base+[0,0,.048],WRIST))
        self.grasp_tcp=self.fk.fk(self.grasp)[0]
        self.lift_path=np.array([np.radians(self.fk.solve_pose(base+[0,0,.048+z],WRIST)) for z in np.linspace(0,.044,9)])
        self.stage_start=t;self.grasp_stage='approach';self.history=[(age,q[:7].copy())]

    def change_grasp_stage(self,stage,t):
        self.grasp_stage=stage;self.stage_start=t

    def grasp_action(self,task,q,t,age):
        if self.abort_time is not None:return np.r_[self.home,np.zeros(4)],'Aborting unconfirmed grasp'
        if self.grasp_stage=='initialize':self.initialize_grasp(task,q,t,age)
        if self.abort_time is not None:return np.r_[self.home,np.zeros(4)],'Aborting unconfirmed grasp'
        elapsed=t-self.stage_start;grip=np.pi/4;stage=self.grasp_stage
        if stage=='approach':
            arm=path_at([self.home,self.grasp],ease(elapsed/1.35));grip=0.
            if elapsed>=1.35:self.change_grasp_stage('close',t)
        elif stage=='close':
            arm=self.grasp.copy();grip*=ease(elapsed/.4)
            if elapsed>=.4:
                self.profile['grasp_mark_s']=age;self.change_grasp_stage('lift',t)
        elif stage=='lift':
            arm=path_at(self.lift_path,ease(elapsed/1.0))
            if elapsed>=1.:self.change_grasp_stage('verify',t)
        elif stage=='verify':
            arm=self.lift_path[-1].copy();neck,score=locate(task,self.saved['source_bank'],tracking=True,expected_uv=self.expected_lift_neck)
            moved=self.initial_neck[1]-neck[1]>25.
            follows=np.linalg.norm(neck-self.expected_lift_neck)<12.
            gap=self.saved['pour_primitive']['grasp_closure']['min_deficit_rad']
            obstructed=held_gripper(q,gap)
            arm_lift=float(self.fk.fk(q[:7])[0][2]-self.grasp_tcp[2])
            confidence=.23
            self.grasp_check=dict(neck_uv=neck.tolist(),expected_uv=self.expected_lift_neck.tolist(),score=score,
                moved=bool(moved),follows=bool(follows),finger_obstruction=bool(obstructed),arm_lift_m=arm_lift)
            if self.threshold.get('grasp_visual_window',0):
                physical=bool(obstructed and arm_lift>.035)
                self.physical_votes=self.physical_votes+1 if physical else 0
                self.visual_votes=(self.visual_votes+[bool(score>confidence and moved and follows)])[-self.threshold['grasp_visual_window']:] if physical else []
                self.confirm_votes=sum(self.visual_votes) if self.physical_votes>=6 else 0
            else:
                self.confirm_votes=self.confirm_votes+1 if score>confidence and moved and follows and obstructed and arm_lift>.035 else 0
            if self.confirm_votes>=6:
                self.grasp_confirmed=True;self.qlift=q[:7].copy();self.lift_rotation=self.fk.fk(self.qlift)[1]
                self.profile['lift_mark_s']=age;self.grasp_stage='ready'
            elif elapsed>self.threshold.get('verify_seconds',.65):self.change_grasp_stage('retry_lower',t)
        elif stage=='retry_lower':
            arm=path_at(self.lift_path[::-1],ease(elapsed/.7))
            if elapsed>=.7:self.change_grasp_stage('retry_open',t)
        elif stage=='retry_open':
            arm=self.grasp.copy();grip*=1-ease(elapsed/.3)
            if elapsed>=.3:self.change_grasp_stage('retry_retreat',t)
        elif stage=='retry_retreat':
            arm=path_at([self.grasp,self.home],ease(elapsed/.6));grip=0.
            if elapsed>=.6:
                if self.attempt<self.threshold['max_attempts']:self.grasp_stage='initialize'
                else:self.abort_time=t;self.stop_latched='grasp_unconfirmed'
        else:arm=q[:7].copy()
        return np.r_[arm,np.full(4,grip)],'Visual grasp: '+stage

    def latch(self,reason,t,robot):
        self.stop_latched=reason;robot.reason=reason;robot.stop_time=t
        self.terminal=TerminalPrimitive(self.history,t,self.profile)
        self.terminal.return_path=monotone_return_path(self.outbound,self.pour_phase,self.history[-1][1],self.qlift)

    @torch.inference_mode()
    def act(self,observation,robot,t):
        if self.start is None:self.start=t;Hardware.hardware(self,robot)
        age=t-self.start;q=observation['joint_pos'];v=observation['joint_vel'];self.history.append((age,q[:7].copy()))
        task=decode(observation['task_encoded'],'jpeg95');wrist=decode(observation['wrist_encoded'],'jpeg95')
        geo=cnn=flow=empty=present=tilt=0.;mass=0.;settled_prediction=None
        if not self.grasp_confirmed and self.terminal is None:
            targets,phase=self.grasp_action(task,q,t,age)
            if self.abort_time is not None:robot.reason=self.stop_latched;robot.stop_time=self.abort_time
        else:
            if not self.outbound or self.pour_phase>self.outbound[-1][0]+.3:
                self.outbound.append((self.pour_phase,q[:7].copy()))
            normalized=cup_crop(task,self.cup_xy,self.cal);a=torch.as_tensor(normalized,device='cuda')[None]
            cnn=float(self.cup_model(a)[0])+self.saved['cup_bias'];features,_,n=surface_features(a);mass=float(n[0])
            geom=self.saved['surface_geometry'];coef=torch.as_tensor(geom['coef'],device='cuda')
            geo=float(features[0,:geom['dimensions']]@coef)+geom['bias'] if mass>1. else 0.
            geo=float(np.clip(geo,0,1));cnn=float(np.clip(cnn,0,1))
            self.filtered=cnn if self.filtered is None else .6*self.filtered+.4*cnn
            self.level_history.append((t,geo,cnn,self.filtered))
            if 'settled_regressor' in self.saved:
                from pour_settled_model import features,predict
                settled_prediction=float(predict(features(a,a.new_tensor([cnn],dtype=torch.float32),self.cup_xy),self.saved['settled_regressor'])[0])
                self.settled_history.append((t,settled_prediction))
            if len(self.level_history)>=9:
                newer=np.median([x[3] for x in self.level_history[-3:]]);older=np.median([x[3] for x in self.level_history[-9:-6]])
                flow=float(np.clip((newer-older)/.3,0,.45))
            c,s,w=crops(task,wrist);p=self.detector(*[torch.as_tensor(x,device='cuda')[None] for x in (c,s,w)])
            empty=float(p['empty_logit'][0].sigmoid());present=float(p['present_logit'][0].sigmoid())
            rotation=self.fk.fk(q[:7])[1];up=rotation@self.lift_rotation.T@np.array([0.,0.,1.])
            tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
            is_empty=empty>=self.threshold['empty_probability'] and present>=.7 and tilt>=95 and flow*72.8<3.
            self.empty_votes=self.empty_votes+1 if is_empty else 0;self.lost_votes=self.lost_votes+1 if present<.05 else 0
            cap=self.saved['pour_primitive']['phase_cap_deg']
            if self.pour_phase>=cap-.25 and self.cap_start is None:self.cap_start=t
            slow=False
            if 'slow_empty_probability' in self.threshold:
                gap=self.saved['pour_primitive']['grasp_closure']['min_deficit_rad']
                slow=slow_empty_evidence(self.threshold,self.pour_phase,cap,tilt,None if self.cap_start is None else t-self.cap_start,
                    held_gripper(q,gap),present,empty,flow*72.8)
            self.slow_empty_votes=self.slow_empty_votes+1 if slow else 0
            if self.stop_latched is None:
                if self.empty_votes>=12 or (slow and self.slow_empty_votes>=self.threshold['slow_empty_votes']):self.latch('bottle_empty',t,robot)
                elif self.lost_votes>=8:self.latch('source_tracking_lost',t,robot)
            if self.terminal is not None:targets,phase=self.terminal.act(t)
            else:
                if self.fill_mode=='brake':
                    self.pour_phase=max(80.,self.pour_phase-90*.05)
                    if self.pour_phase<=80.:
                        self.fill_mode='settle';self.pause_start=t
                elif self.fill_mode=='settle':
                    self.pour_phase=80.
                    if t-self.pause_start>=self.threshold['settle_seconds']:
                        last=[x for x in self.level_history if x[0]>=self.pause_start+.35][-6:]
                        other=float(np.median([x[1] for x in last]))
                        if 'settled_regressor' in self.saved:
                            from pour_settled_model import assess
                            levels=[x[1] for x in self.settled_history if x[0]>=self.pause_start+.35][-6:]
                            settled,stable,full=assess(levels,other,self.threshold['full_settled_target'],
                                self.threshold['settled_stability_std'],self.threshold['settled_geometry_tolerance'],mass)
                        else:
                            settled=float(np.quantile([x[2] for x in last],.25))
                            stable=np.std([x[2] for x in last])<.012
                            full=settled>=self.threshold['full_settled_target'] and abs(other-settled)<.06 and stable and mass>1.
                        self.full_votes=self.full_votes+1 if full else 0
                        if self.full_votes>=3:self.latch('cup_near_full',t,robot)
                        elif not full and settled>=self.threshold['full_settled_target']:
                            if t-self.pause_start>2.5:self.latch('level_unconfirmed',t,robot)
                        elif not full and (stable or t-self.pause_start>1.6):
                            if self.pause_count>=self.threshold.get('max_pauses',3):self.latch('fill_unconfirmed',t,robot)
                            else:
                                self.fill_mode='trim'
                                self.trim_goal=topup_goal(settled,self.threshold['full_settled_target'],self.threshold.get('trim_increment',.015))
                                self.trim_min_phase=self.threshold.get('trim_min_phase',max(95.,self.peak_phase-.5))
                                if self.threshold.get('adaptive_trim'):
                                    from pour_settled_model import topup_cap
                                    self.trim_cap=topup_cap(self.peak_phase,self.threshold['trim_extra_deg'],self.threshold['trim_hard_cap_deg'])
                                self.trim_start=None
                                self.trim_entered=t
                else:
                    movement=self.saved['pour_primitive']
                    if self.fill_mode=='coarse':rate=movement['fast_rate_deg_s'] if self.pour_phase<82 else movement['slow_rate_deg_s']
                    elif self.threshold.get('discrete_topup'):
                        rate=self.threshold['trim_fast_rate_deg_s']
                    elif self.threshold.get('adaptive_trim'):
                        rate=self.threshold['trim_fast_rate_deg_s'] if self.pour_phase<self.trim_cap-2. else self.threshold['trim_creep_rate_deg_s']
                    else:rate=20. if self.pour_phase<self.peak_phase-2 else 2.
                    if self.filtered>.65 and flow*72.8>6. and not (self.fill_mode=='trim' and self.threshold.get('discrete_topup')):rate=0.
                    self.pour_phase=min(movement['phase_cap_deg'],self.pour_phase+.05*rate)
                    if self.fill_mode=='trim':
                        self.pour_phase=min(self.trim_cap,self.pour_phase)
                        if self.pour_phase>=self.trim_cap-.5 and self.trim_start is None:self.trim_start=t
                    self.peak_phase=max(self.peak_phase,self.pour_phase)
                    goal=self.threshold['coarse_brake_target'] if self.fill_mode=='coarse' else self.trim_goal
                    eligible=self.fill_mode=='coarse' or self.pour_phase>=self.trim_min_phase
                    if self.fill_mode=='trim' and self.threshold.get('adaptive_trim'):
                        from pour_settled_model import topup_feedback_eligible
                        eligible=topup_feedback_eligible(self.pour_phase,self.trim_cap)
                    timed_brake=self.fill_mode=='trim' and (trim_expired(self.trim_start,t,self.threshold.get('trim_seconds',.1)) or
                        trim_expired(self.trim_entered,t,self.threshold.get('trim_total_seconds',.9)))
                    if self.fill_mode=='trim' and self.threshold.get('discrete_topup'):
                        from pour_settled_model import pulse_finished
                        timed_brake=pulse_finished(self.pour_phase,self.trim_cap,t-self.trim_entered,self.threshold['trim_total_seconds'])
                        eligible=False
                    feedback=settled_prediction if self.fill_mode=='trim' and settled_prediction is not None else self.filtered
                    if timed_brake or (eligible and feedback>.60 and feedback+flow*self.threshold['runoff_seconds']>=goal):
                        self.fill_mode='brake';self.pause_count+=1;self.full_votes=0
                if self.terminal is not None:targets,phase=self.terminal.act(t)
                else:
                    arm=target(self.saved['pour_primitive'],self.qlift,self.cup_xy,self.pour_phase)
                    targets=np.r_[arm,np.full(4,np.pi/4)];phase='Pour control: '+self.fill_mode
        bounded=np.clip(targets-q,-np.r_[[.12]*7,[.85]*4],np.r_[[.12]*7,[.85]*4]);command=q+bounded
        limited=np.clip(command[:7],self.arm_limits[0],self.arm_limits[1]);clamped=not np.allclose(command[:7],limited) or not np.allclose(targets-q,bounded)
        command[:7]=limited;command[7:]=np.clip(command[7:],0,np.pi/4)
        self.clamps+=int(clamped);self.steps+=1;robot.phase='Visual guard: '+phase
        released=np.max(abs(q[7:]))<.1 and np.max(abs(v))<.08
        holding=(self.terminal is not None and self.terminal.stage==4) or (self.abort_time is not None and t-self.abort_time>.5)
        self.done_votes=self.done_votes+1 if holding and released else 0
        self.diagnostics.append(dict(t=t,mode=phase,stop_latched=self.stop_latched,grasp_stage=self.grasp_stage,grasp_attempt=self.attempt,
            visual_grasp_confirmed=self.grasp_confirmed,confirmation_votes=self.confirm_votes,source_xy=None if self.source_xy is None else self.source_xy.tolist(),
            grasp_check=self.grasp_check,
            cup_xy=None if self.cup_xy is None else self.cup_xy.tolist(),cup_predicted_fraction=cnn,surface_geometry_fraction=geo,
            settled_prediction=settled_prediction,source_localization_score=getattr(self,'source_score',None),cup_localization_score=getattr(self,'cup_score',None),
            visual_flow_fraction_s=flow,empty_probability=empty,source_present_probability=present,estimated_tilt_from_joints=tilt,
            physical_confirmation_votes=self.physical_votes,visual_confirmation_window=self.visual_votes,
            slow_empty_votes=self.slow_empty_votes,cap_dwell_seconds=None if self.cap_start is None else t-self.cap_start,
            own_pour_phase_deg=self.pour_phase,fill_mode=self.fill_mode,pause_count=self.pause_count,trim_goal=self.trim_goal,
            trim_cap_deg=self.trim_cap,
            delta_clamped=bool(clamped),done_probability=float(self.done_votes>=10)))
        return np.degrees(command)

    def complete(self):return self.done_votes>=10
