"""Joint-driven BG2 reference pour. The bottle is free throughout the episode."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from bottle_pour import Pour, GRASP_Z, START_X, pose, ease

# Cup frame (+X toward the stream) expressed in robot base coordinates.
CUP_WORLD = np.array([.49, -.15, 1.04])
FRAME = Rotation.from_euler('z', 90, degrees=True).as_matrix()
WRIST = Rotation.from_euler('y', -90, degrees=True).as_matrix()
TCP = np.array([0., 0., -.202])
ROOT = '/World/Robot/ffw_bg2_follower'
TILT_START = 3.3
FINGERS = ('gripper_r_rh_p12_rn_r2', 'gripper_r_rh_p12_rn_l2')


class ArmPath:
    """Solve wrist poses within the seven USD joint limits, in path order."""
    def __init__(self, offsets, axes, limits, shoulder, cup_world=CUP_WORLD, start_xy=None, mouth_z=.054, wide_geometry=False):
        self.cup_world = np.asarray(cup_world)
        self.start_xy = start_xy
        self.mouth_z = mouth_z
        self.wide_geometry = wide_geometry
        self.offsets = np.asarray(offsets)
        self.axes = np.eye(3)[['XYZ'.index(a) for a in axes]]
        self.limits = np.radians(limits).T
        self.shoulder = np.asarray(shoulder)
        self.q = np.array([-.25, -.45, -.15, -1.4, 0., .25, 0.])

    def fk(self, q):
        xyz, rot = self.shoulder.copy(), np.eye(3)
        for delta, axis, angle in zip(self.offsets, self.axes, q):
            xyz = xyz + rot @ delta
            rot = rot @ Rotation.from_rotvec(axis*angle).as_matrix()
        return xyz + rot @ TCP, rot

    def solve(self, angle, lift=1., retreat=0.):
        if self.wide_geometry:
            from pour_wide_geometry import calibrated_pose
            base, rot = calibrated_pose(angle, lift, self.start_xy, self.mouth_z)
        else:
            base, rot = pose(angle, lift, self.start_xy, self.mouth_z)
            base[2] += .018*lift*(1-ease(angle/60))
        target = self.cup_world + FRAME @ (base + rot @ [0.,0.,GRASP_Z]) - [retreat,0.,0.]
        orientation = FRAME @ rot @ FRAME.T @ WRIST
        return self.solve_pose(target, orientation)

    def solve_pose(self, target, orientation):
        """Solve an observed-grasp TCP pose without prescribing bottle motion."""
        previous = self.q
        def residual(q):
            xyz, wrist = self.fk(q)
            return np.r_[5*(xyz-target), Rotation.from_matrix(orientation @ wrist.T).as_rotvec(), .003*(q-previous)]
        solution = least_squares(residual, previous, bounds=self.limits, max_nfev=85)
        error = residual(solution.x)
        if max(abs(error[:6])) > .012:
            raise ValueError(f'Unreachable wrist target: {error[:6]}')
        self.q = solution.x
        return np.degrees(self.q)


def build_path(stage, spec, include_planner=False):
    from pxr import UsdPhysics
    joints=[UsdPhysics.RevoluteJoint.Get(stage, ROOT+'/joints/arm_r_joint'+str(i)) for i in range(1,8)]
    lift=UsdPhysics.PrismaticJoint.Get(stage,ROOT+'/joints/lift_joint')
    planner=ArmPath([j.GetLocalPos0Attr().Get() for j in joints],
                    [j.GetAxisAttr().Get() for j in joints],
                    [(j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()) for j in joints],
                    lift.GetLocalPos0Attr().Get(), spec.get('cup_world',CUP_WORLD),
                    spec.get('source_xy'),spec.get('mouth_z',.054),wide_geometry=spec.get('protocol') in (17,18))
    nominal=ArmPath(planner.offsets,[j.GetAxisAttr().Get() for j in joints],
                    [(j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()) for j in joints],
                    lift.GetLocalPos0Attr().Get())
    home=nominal.solve(0.,0.,.07)
    # Wider fixtures can occupy the old low ready-hand position. Raise the
    # shared ready pose before simulation begins; the bottle stays dynamic.
    # This is fixed robot setup, never derived from live object poses.
    home_clearance=spec.get('home_clearance_m', .10 if spec.get('protocol') in (17,18) else 0.)
    if home_clearance:
        tcp,rotation=nominal.fk(np.radians(home))
        home=nominal.solve_pose(tcp+[0.,0.,home_clearance],rotation)
    grasp=planner.solve(0.,0.)
    approach=np.linspace(home,grasp,13)
    lift_path=np.array([planner.solve(0.,x) for x in np.linspace(0.,1.,9)])
    angles=np.arange(0.,166.,3.)
    tilt=np.array([planner.solve(a) for a in angles])
    if spec.get('approach_clearance_m'):
        from pour_workspace_path import clearance_approach
        approach=np.degrees(clearance_approach(planner,np.radians(home),np.radians(grasp),
            clearance_m=spec['approach_clearance_m'],level_transit=spec.get('protocol') in (17,18)))
    return (approach,lift_path,tilt,planner) if include_planner else (approach,lift_path,tilt)


def interpolate(path, fraction):
    x=np.clip(fraction,0.,1.)*(len(path)-1)
    i=min(int(x),len(path)-2)
    return path[i]+(x-i)*(path[i+1]-path[i])


class RobotPour(Pour):
    def __init__(self, stage, spec, bottle_glass, cup_glass):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
        from omni.physx import get_physx_simulation_interface
        super().__init__(stage,spec,bottle_glass,cup_glass)
        self.stage=stage
        self.cup_world=np.asarray(spec.get('cup_world',CUP_WORLD))
        self.start_xy=np.asarray(spec.get('source_xy',[START_X,0.]))
        self.last_observation=None
        self.ever_table_particles=None
        self.bottle_prim=stage.GetPrimAtPath('/World/Bottle')
        UsdPhysics.RigidBodyAPI(self.bottle_prim).GetKinematicEnabledAttr().Set(False)
        # Only initial fixture placement writes the bottle transform.
        self.position.Set(Gf.Vec3d(*(self.cup_world + FRAME @ [*self.start_xy,.0002])))
        self.points[:,:2] += self.start_xy - [START_X,0.]
        self.points=(self.points @ FRAME.T+self.cup_world+[0.,0.,.0002]).astype(np.float32)
        self.initial_bottle_count=len(self.points)
        if spec.get('cup_initial_ml',0)>0:
            spacing=spec['spacing']
            radius=.020-2*spacing
            n=int(radius/spacing)
            cross=[(i*spacing,j*spacing) for i in range(-n,n+1) for j in range(-n,n+1)
                   if (i*i+j*j)*spacing**2<radius**2]
            count=round(spec['cup_initial_ml']/(spacing**3*1e6))
            cup_points=np.array([[*cross[i%len(cross)],.004+2*spacing+(i//len(cross))*spacing]
                                 for i in range(count)],dtype=np.float32)
            if 'cup_height_scale' in spec or 'cup_diameter_scale' in spec:
                from pour_vessel_geometry import initial_cup_points
                cup_points=initial_cup_points(spec)
            if spec.get('calibration_version') in (1,2,3):
                from pour_finish_calibration import cup_points as calibration_points
                cup_points=calibration_points(spec)
                assert abs(len(cup_points)*spacing**3*1e6-spec['cup_initial_ml'])<1e-6
            self.points=np.concatenate([self.points,(cup_points @ FRAME.T+self.cup_world).astype(np.float32)])

        cup=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Cup'))
        cup.GetOrderedXformOps()[0].Set(Gf.Vec3d(*self.cup_world))
        robot=stage.DefinePrim('/World/Robot','Xform')
        robot.GetReferences().AddReference('/workspace/assets/FFW_BG2_without_mimic.usd')
        UsdPhysics.Joint.Get(stage,ROOT+'/world/FixedJoint').CreateJointEnabledAttr(False)
        grip=UsdShade.Material.Define(stage,'/World/FingerContact')
        api=UsdPhysics.MaterialAPI.Apply(grip.GetPrim())
        api.CreateStaticFrictionAttr(1.15)
        api.CreateDynamicFrictionAttr(1.)
        api.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(grip.GetPrim()).CreateFrictionCombineModeAttr('max')
        for prim in Usd.PrimRange(robot):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr('convexHull')
            if prim.HasAPI(UsdPhysics.CollisionAPI) and any('/'+name+'/' in str(prim.GetPath()) for name in FINGERS):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(grip,materialPurpose='physics')
                phys=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                phys.CreateContactOffsetAttr(.001)
                phys.CreateRestOffsetAttr(0.)
        self.approach,self.lift_path,self.tilt,self.path_planner=build_path(stage,spec,include_planner=True)
        self.centered_return=None
        self.scene_evidence = dict(
            bottle_mass_kg=.18, bottle_gravity_enabled=True, bottle_kinematic=False,
            bottle_attachment_joints=[], robot_arm_gravity_compensation=True,
            finger_static_friction=1.15, finger_dynamic_friction=1.,
            arm_control='Joint position drives with native actuator torque limits',
            observation_source='Simulator particle inventory and fill height',
            home_clearance_m=spec.get('home_clearance_m', .10 if spec.get('protocol') in (17,18) else 0.),
            dof_names=['arm_r_joint'+str(i) for i in range(1,8)]+
                      ['gripper_r_joint'+str(i) for i in range(1,5)],
            cup_frame_world=self.cup_world.tolist(), cup_frame_rotation=FRAME.tolist())

        self.drives=[]
        self.states=[]
        for i in range(11):
            arm=i<7
            name=('arm_r_joint'+str(i+1)) if arm else ('gripper_r_joint'+str(i-6))
            prim=stage.GetPrimAtPath(ROOT+'/joints/'+name)
            drive=UsdPhysics.DriveAPI(prim,'angular')
            initial=float(self.approach[0,i]) if arm else 0.
            prim.GetAttribute('state:angular:physics:position').Set(initial)
            prim.GetAttribute('state:angular:physics:velocity').Set(0.)
            drive.CreateStiffnessAttr((600 if i<6 else 200)*math.pi/180 if arm else .50)
            drive.CreateDampingAttr((25 if i<6 else 4)*math.pi/180 if arm else .018)
            drive.CreateMaxForceAttr(([61.4,61.4,31.7,31.7,31.7,31.7,5.1][i]) if arm else 3.)
            self.drives.append(drive.CreateTargetPositionAttr(initial))
            self.states.append(prim.GetAttribute('state:angular:physics:position'))
        PhysxSchema.PhysxContactReportAPI.Apply(self.bottle_prim).CreateThresholdAttr(0.)
        self.contacts=set()
        self.sub=get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)
        self.held_frames=0
        self.verified=False
        self.next_control=0.
        self.command_angle=0.
        self.command=np.r_[self.approach[0],np.zeros(4)]
        self.stop_command_angle=0.

    def on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        for header in headers:
            if not header.num_contact_data: continue
            actors=[str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.actor0,header.actor1)]
            if '/World/Bottle' in actors:
                self.contacts.update(path.rsplit('/',1)[-1] for path in actors if path!='/World/Bottle')

    def drive(self,t):
        if self.spec.get('calibration_only'):
            self.command=np.r_[self.approach[0],np.zeros(4)]
            self.phase='Stationary liquid-level calibration'
            for drive,target in zip(self.drives,self.command):drive.Set(float(target))
            return
        if t+1e-8<self.next_control: return
        self.next_control=t+self.spec.get('control_period',.01)
        if self.stop_time is None:
            hold = self.spec.get('reference_hold_seconds', 0.)
            extension = self.spec.get('reference_approach_extension_s', 0.)
            # Keep an initial stationary calibration window and extend only the
            # approach. Stop/return clocks always remain actual simulation time.
            u = max(0., t - hold)
            if .25 <= u <= 1. + extension:
                t = .25 + .75 * (u - .25) / (.75 + extension)
            elif u > 1. + extension:
                t = u - extension
            else:
                t = u
        closure=ease((t-1.)/.65)
        if t<1.:
            arm=interpolate(self.approach,ease((t-.25)/.75))
            self.phase='Approaching bottle'
        elif t<1.85:
            arm=self.approach[-1]
            self.phase='Closing fingers'
        elif t<TILT_START:
            arm=interpolate(self.lift_path,ease((t-1.85)/.65))
            self.phase='Lifting and checking grip'
        elif self.stop_time is None:
            if self.spec.get('gentle') and self.verified:
                row=self.last_observation or {}
                flow=row.get('source_outflow_ml_s',0.)
                fill=row.get('cup_inventory_fraction',0.)
                limit=6. if fill>.65 else 12.
                rate=30. if self.command_angle<82. else (self.spec.get('expert_slow_rate_deg_s',10.) if flow<limit else 0.)
                self.command_angle=min(165.,self.command_angle+rate*self.spec.get('control_period',.05))
            else:
                self.command_angle=min(165.,max(0.,(t-TILT_START)*33.)) if self.verified else 0.
            arm=interpolate(self.tilt,self.command_angle/165.)
            self.phase='Pouring from gripper' if self.verified else 'Grasp check failed'
        else:
            elapsed=max(0.,t-self.stop_time-self.reference_empty_dwell())
            self.command_angle=self.stop_command_angle*(1-ease(elapsed/.85))
            arm=interpolate(self.tilt,self.command_angle/165.)
            self.phase=('Holding empty source for supervision' if t<self.stop_time+self.reference_empty_dwell() else 'Returning upright')
            if elapsed>.95:
                arm=interpolate(self.lift_path,1-ease((elapsed-.95)/.65))
                self.phase='Setting bottle down'
            if elapsed>1.8:
                closure=1-ease((elapsed-1.8)/.35)
                arm=interpolate(self.approach,1-ease((elapsed-2.2)/.5))
                self.phase='Releasing bottle'
            if self.spec.get('protocol') in (17,18) and self.verified and self.stop_command_angle>=70.:
                from pour_wide_motion import centered_return_path,timed_target
                if self.centered_return is None:
                    source=self.cup_world+FRAME@np.r_[self.start_xy,0.]
                    current=np.radians([float(s.Get()) for s in self.states[:7]])
                    self.centered_return=centered_return_path(self.path_planner,current,self.stop_command_angle,
                        source[:2],self.cup_world[:2],self.cup_world[2],np.radians(self.lift_path[-1]),self.spec.get('mouth_z',.054))
                times,path=self.centered_return;end=float(times[-1])
                self.command_angle=self.stop_command_angle*(1-ease(elapsed/1.1))
                arm=np.degrees(timed_target(times,path,elapsed));closure=1.
                self.phase=('Holding empty source for supervision' if t<self.stop_time+self.reference_empty_dwell() else 'Returning upright')
                if elapsed>end:
                    arm=interpolate(self.lift_path,1-ease((elapsed-end)/.7));self.phase='Setting bottle down'
                if elapsed>end+.8:
                    closure=1-ease((elapsed-end-.8)/.35);self.phase='Releasing bottle'
                if elapsed>end+1.25:
                    arm=interpolate(self.approach,1-ease((elapsed-end-1.25)/.7))
        self.command=np.r_[arm,np.full(4,45.*closure)]
        self.teacher_command=self.command.copy()
        if self.spec.get('noisy_expert'):
            from pour_perturbations import approach_noise
            self.command[:7]+=approach_noise(t,self.spec['noise_seed'],self.spec['arm_noise_deg'])
        for drive,target in zip(self.drives,self.command):
            drive.Set(float(target))

    def observe(self,p,v,t):
        from pxr import Usd,UsdGeom
        cache=UsdGeom.XformCache(Usd.TimeCode.Default())
        transform=np.array(cache.GetLocalToWorldTransform(self.bottle_prim)).T
        self.base=FRAME.T @ (transform[:3,3]-self.cup_world)
        self.rotation=FRAME.T @ transform[:3,:3]
        self.angle=math.degrees(math.atan2(self.rotation[0,2],self.rotation[2,2]))
        contacts=sorted(self.contacts)
        self.contacts.clear()
        both=all(name in contacts for name in FINGERS)
        reference_delay = self.spec.get('reference_hold_seconds', 0.) + self.spec.get('reference_approach_extension_s', 0.)
        if (self.spec.get('policy_evaluation') or 2.55+reference_delay<=t<=TILT_START+reference_delay) and both and self.base[2]>.030 and abs(self.angle)<8:
            self.held_frames+=1
        self.verified |= self.held_frames>=6
        if not self.spec.get('policy_evaluation') and t>=TILT_START+reference_delay and not self.verified and self.stop_time is None:
            self.reason='grasp_failed'
            self.stop_time=t
        stopped=self.stop_time is not None
        row=super().observe((p-self.cup_world) @ FRAME,v @ FRAME,t)
        self.last_observation=row
        local=(p-self.cup_world) @ FRAME
        if self.spec.get('record_particle_diagnostics'):
            from bottle_pour import vessel_mask,BOTTLE
            outside=~(vessel_mask(local,self.cup_profile)|vessel_mask((local-self.base)@self.rotation,BOTTLE))
            ids=np.flatnonzero(outside)
            row['diagnostic_outside_particles']=[dict(index=int(i),origin='bottle' if i<self.initial_bottle_count else 'cup',
                xyz_cup_frame_m=local[i].tolist(),velocity_m_s=(v[i]@FRAME).tolist()) for i in ids[:32]]
            row['diagnostic_outside_count']=len(ids)
        if self.ever_table_particles is None: self.ever_table_particles=np.zeros(len(p),dtype=bool)
        self.ever_table_particles |= (local[:,2]<.0035)
        row['ever_table_ml']=float(self.ever_table_particles.sum()*self.spec['spacing']**3*1e6)
        if self.stop_time is not None and not stopped:
            self.stop_command_angle=self.command_angle
        row.update(bottle_pose_xyz=transform[:3,3].tolist(),
                   bottle_quat_xyzw=Rotation.from_matrix(transform[:3,:3]).as_quat().tolist(),
                   grasp_verified=bool(self.verified),contacts=contacts,both_fingers=both,
                   bottle_base_height_m=float(self.base[2]),joint_targets_deg=self.command.tolist(),
                   joint_positions_deg=[float(s.Get()) for s in self.states],command_angle_deg=self.command_angle)
        return row

    def reference_empty_dwell(self):
        if self.spec.get('policy_evaluation') or not self.verified or self.reason!='bottle_empty':return 0.
        return self.spec.get('reference_empty_dwell_s',1.2 if self.spec.get('protocol') in (17,18) else 0.)

    def complete(self,t):
        if self.spec.get('calibration_only'):return t>=self.spec['max_seconds']
        if self.spec.get('corrective_prefix'): return t>=self.spec.get('prefix_seconds',3.3)
        terminal_s=3.
        if self.spec.get('protocol') in (17,18) and self.verified and self.stop_command_angle>=70.:
            from pour_wide_motion import RETURN_SECONDS
            terminal_s=RETURN_SECONDS+2.1
        return (self.stop_time is not None and t>=self.stop_time+terminal_s+self.reference_empty_dwell()) or t>=self.spec['max_seconds']+.65

    def summary(self,rows):
        result=super().summary(rows)
        hold=[r for r in rows if 2.55<=r['t']<=TILT_START]
        tipping=[r for r in rows if abs(r['bottle_tilt_deg'])>20.]
        good=sum(r['both_fingers'] for r in tipping)/max(1,len(tipping))
        result.update(attribution='articulated BG2 joint drives, dynamic bottle, simulator-state stop controller',
                      grasp_verified=bool(self.verified),bilateral_fraction_during_pour=good,
                      hold_height_m=float(np.median([r['bottle_base_height_m'] for r in hold])) if hold else 0.)
        if not self.verified or good<.8: result['gates'].append('grip_not_sustained')
        if abs(rows[-1]['bottle_tilt_deg'])>8 or abs(rows[-1]['bottle_base_height_m'])>.005:
            result['gates'].append('set_down_failed')
        result['scene'] = self.scene_evidence
        result['spill'] = dict(final_outside_ml=rows[-1]['spilled_ml']+rows[-1]['airborne_ml'],
                              ever_table_ml=max(r.get('ever_table_ml',0.) for r in rows),
                              particle_quantum_ml=self.spec['spacing']**3*1e6)
        result['zero_spill'] = (result['spill']['final_outside_ml']==0 and result['spill']['ever_table_ml']==0)
        if self.spec.get('policy_evaluation'):
            result['attribution']='learned recurrent RGB/joint behavior cloning; no teacher actions or state stop'
            result['scene']['observation_source']='Task RGB, wrist RGB, measured joint positions/velocities'
        if self.spec.get('expert_recovery'):
            result['attribution']='Learned RGB/joint prefix followed by privileged scripted expert recovery; not autonomous success'
            result['scene']['observation_source']='RGB/joints before takeover; measured object pose and particle inventory in training-only expert'
        if self.spec.get('policy_family') in ('visual_stop_hierarchy','visual_guard_hierarchy'):
            result['attribution']='Hierarchical visual imitation: RGB liquid estimator, demonstration-fitted pour/terminal primitives, irreversible stop latch'
            result['scene']['observation_source']='RGB, measured joints/velocities, own motion history; no privileged runtime state'
        if self.spec.get('policy_family')=='visual_guard_hierarchy':
            result['attribution']='Image-localized visual servo grasp/retry, normalized supervised cup model, learned pour primitive, settled-level top-ups, monotone terminal sequence'
            result['scene']['observation_source']='RGB, measured joints/velocities, own history/phase and fixed robot/camera/bench calibration; no runtime object/particle state'
        result['eligible']=not result['gates']
        result['task_success_zero_spill'] = bool(result['eligible'] and result['zero_spill'] and
            (result['expected_stop']!='cup_near_full' or rows[-1]['cup_inventory_fraction']>=.90))
        if self.spec.get('policy_family') in ('visual_stop_hierarchy','visual_guard_hierarchy'):
            f=rows[-1];empty=f['bottle_ml']<=max(.25,.01*self.spec['initial_ml'])
            goal=(f['stop_reason']=='cup_near_full' and f['cup_fill_fraction']>=.90) or (f['stop_reason']=='bottle_empty' and empty)
            result['surface_task_success_zero_spill']=bool(self.verified and good>=.8 and result['zero_spill'] and goal and
                abs(f['bottle_tilt_deg'])<8 and abs(f['bottle_base_height_m'])<=.005)
            result['primary_goal']='measured surface height >=90% rim OR source empty, sustained grasp, upright set-down/release, zero escaped particles'
            result['legacy_inventory_task_success_zero_spill']=result['task_success_zero_spill']
        if self.spec.get('corrective_prefix'):
            bilateral=sum(r['both_fingers'] for r in hold)/max(1,len(hold))
            passed=bool(self.verified and bilateral>=.8 and result['zero_spill'] and
                        rows[-1]['bottle_base_height_m']>.030 and abs(rows[-1]['bottle_tilt_deg'])<8)
            result.update(task_scope='approach_lift_prefix',full_task_success=False,
                prefix_success=passed,eligible=passed,gates=[] if passed else ['corrective_grasp_failed'],
                task_success_zero_spill=False,bilateral_fraction_during_lift=bilateral)
        self.sub=None
        return result
