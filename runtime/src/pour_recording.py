"""Synchronized RGB and joint-action transitions for the native BG2 pouring fixture."""
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom
from PIL import Image

sys.path.append('/workspace/pydeps51')
import h5py
from pour_camera_clock import CAMERA_TOLERANCE_S, camera_time_is_current

CAMERA_SIZE = (384, 384)
DT = .05
NAMES = ['arm_r_joint'+str(i) for i in range(1,8)] + ['gripper_r_joint'+str(i) for i in range(1,5)]
ROOT = '/World/Robot/ffw_bg2_follower'


class PourRecording:
    def __init__(self, stage, folder, spec, sim, robot, preview):
        from isaacsim.sensors.camera import Camera
        self.stage, self.folder, self.sim = stage, folder, sim
        self.robot, self.preview = robot, preview
        self.cameras = {}
        self.camera_sizes = {}
        self.wrist_mount = None
        self.image_encoding=spec.get('image_encoding','raw')
        self.last = None
        self.count = 0
        self.closed = False
        self.capture_retries = 0
        self.max_skew = 0.
        definitions = {
            'task': ('/World/TrainingTaskCamera', [.85,-.55,1.35], [.49,-.22,1.12], [0,0,1], spec.get('task_camera_focal_m', .040)),
            'wrist': ('/World/TrainingWristCamera', [.16,-.07,-.025], [0,.04,-.22], [1,0,0], .007),
        }
        if spec.get('record_overview'):
            # Recording-only spectator: never supplied to either vision model.
            # Preserve the calibrated task/wrist lenses and poses verbatim.
            definitions['overview'] = ('/World/RecordingOverviewCamera',
                [1.0,-.73,1.65], [.49,-.22,1.12], [0,0,1], .024)
        if spec.get('record_cup_detail'):
            detail_resolution=spec.get('cup_detail_resolution',768)
            detail_focal=spec.get('cup_detail_focal_m',.045)
            assert detail_resolution in (768,1024) and detail_focal in (.030,.045)
            definitions['cup_detail'] = ('/World/CupDetailCamera',
                [.85,-.55,1.35], [.495,-.225,1.060], [0,0,1], detail_focal)
        for name, (path, eye, target, up, focal) in definitions.items():
            size = (detail_resolution,detail_resolution) if name == 'cup_detail' else CAMERA_SIZE
            self.camera_sizes[name] = size
            camera = Camera(prim_path=path, name='pour_'+name, frequency=-1,
                            resolution=size, annotator_device='cpu')
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
            if name == 'wrist':
                self.wrist_mount = matrix
                parent = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                    stage.GetPrimAtPath(ROOT+'/right_arm/arm_r_link7'))
                matrix = matrix * parent
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(matrix)
            # Native USD lens attributes use millimetre-scale values. Keep the
            # focal/aperture ratio calibrated while avoiding tiny-value renderer clamps.
            camera.prim.GetAttribute('focalLength').Set(focal*1000.)
            camera.prim.GetAttribute('horizontalAperture').Set(20.955)
            camera.prim.GetAttribute('verticalAperture').Set(20.955)
            camera.set_clipping_range(.005, 5.)
            camera.initialize()
            self.cameras[name] = camera

        for _ in range(12):
            sim.render()
        for _ in range(round(DT/sim.get_physics_dt())):
            sim.step(render=False)
        self.update_camera_poses()
        sim.render()
        self.last = self.capture(float(sim.current_time))
        self.path = folder/'episode.partial.hdf5'
        self.file = h5py.File(self.path, 'w')
        self.demo = self.file.create_group('data/demo_0')
        self.datasets = {}
        for group in ('obs','next_obs'):
            for name in self.cameras:
                if self.image_encoding=='jpeg95':
                    self.create(group+'/'+name+'_image', (), h5py.vlen_dtype(np.uint8))
                else:
                    self.create(group+'/'+name+'_image', self.camera_sizes[name]+(3,), np.uint8)
                self.create(group+'/'+name+'_time', (), np.float64)
                self.create(group+'/'+name+'_reference', (2,), np.int64)
                self.create(group+'/'+name+'_world_T_camera', (4,4), np.float64)
            self.create(group+'/joint_pos', (11,), np.float32)
            self.create(group+'/joint_vel', (11,), np.float32)
        self.create('actions', (11,), np.float32)
        if spec.get('expert_recovery'):
            self.create('expert_supervision', (), np.uint8)
        if spec.get('noisy_expert') and not spec.get('policy_evaluation'):
            self.create('teacher_actions',(11,),np.float32)
        self.create('action_time', (), np.float64)
        self.create('time', (), np.float64)
        self.create('next_time', (), np.float64)
        self.create('rewards', (), np.float32)
        self.create('dones', (), np.uint8)
        self.create('ground_truth/bottle_pose', (7,), np.float64)
        for name in ('cup_ml','bottle_ml','spilled_ml','airborne_ml','cup_inventory_fraction','bottle_tilt_deg','command_angle_deg'):
            self.create('ground_truth/'+name, (), np.float64)
        self.create('ground_truth/grasp_verified', (), np.uint8)
        self.create('ground_truth/stop_command', (), np.uint8)
        self.create('ground_truth/stop_reason', (), h5py.string_dtype())
        self.create('ground_truth/phase', (), h5py.string_dtype())
        self.file.attrs['status'] = 'recording'
        self.file.attrs['schema'] = 'aiworker_rgb_joint_transitions_v2'
        if spec.get('noisy_expert') and not spec.get('policy_evaluation'):
            self.file.attrs['schema']='aiworker_rgb_joint_transitions_v3_teacher_labels'
            self.file.attrs['teacher_action_semantics']='Clean expert joint target for obs(t), radians. actions records the actually executed perturbed target. Labels are not action replay.'
        self.file.attrs['image_encoding']=self.image_encoding
        self.file.attrs['evidence_kind'] = ('learned_visual_policy' if spec.get('policy_evaluation') else
                                          'scripted_action_replay' if spec.get('replay_episode') else
                                          'noisy_expert_reference' if spec.get('noisy_expert') else 'scripted_reference')
        if spec.get('expert_recovery'):
            self.file.attrs['schema']='aiworker_rgb_joint_transitions_v4_expert_takeover'
            self.file.attrs['evidence_kind']='policy_prefix_expert_recovery'
            self.file.attrs['supervision_semantics']='expert_supervision=1 only for actual expert-issued actions; unsupervised policy prefix supplies history only'
        self.file.attrs['camera_rig'] = 'A_task_wrist_v4'
        if spec.get('record_overview'):
            self.file.attrs['camera_rig'] = 'A_task_wrist_v4_with_spectator_v1'
            self.file.attrs['policy_camera_names'] = json.dumps(['task','wrist'])
            self.file.attrs['spectator_camera_names'] = json.dumps(['overview'])
        if spec.get('task_camera_focal_m', .040) != .040:
            self.file.attrs['camera_rig'] = 'A_task_wide_wrist_v5' + ('_with_spectator_v1' if spec.get('record_overview') else '')
            self.file.attrs['task_camera_focal_m'] = spec['task_camera_focal_m']
        self.file.attrs['camera_visual_review'] = 'pending'
        if spec.get('record_cup_detail'):
            self.file.attrs['camera_rig'] = 'A_task_wrist_cup_detail_v6'
            self.file.attrs['policy_camera_names'] = json.dumps(['task','wrist','cup_detail'])
            self.file.attrs['cup_detail_resolution_wh'] = self.camera_sizes['cup_detail']
        self.file.attrs['joint_names'] = json.dumps(NAMES)
        self.file.attrs['control_dt'] = DT
        self.file.attrs['action_semantics'] = 'Absolute joint position targets in radians, held for the following 0.05 simulation seconds'
        self.file.attrs['camera_timestamp_tolerance_s'] = CAMERA_TOLERANCE_S
        self.file.attrs['time_semantics'] = ('Raw camera rendering_time is preserved. Camera/physics skew must be below '
            '100 microseconds (1/25 of the 2.5 ms physics step and 1/500 of the control interval). '
            'This bound admits numerical clock roundoff, not an old physics frame.')
        self.file.attrs['camera_pose_convention'] = 'world_T_camera in USD camera axes (+Y up, -Z forward), metres'
        self.file.attrs['variant'] = 'A'
        self.file.attrs['task_scope']='approach_lift_prefix' if spec.get('corrective_prefix') else 'full_pour'
        if spec.get('calibration_only'):
            self.file.attrs['task_scope']='stationary_perception_calibration'
            self.file.attrs['evidence_kind']='stationary_perception_calibration'
            if spec.get('calibration_motion'):
                self.file.attrs['task_scope']='moving_perception_calibration'
                self.file.attrs['evidence_kind']='authored_moving_perception_calibration'
        self.file.attrs['spec'] = json.dumps(spec)
        self.file['data'].attrs['env_args'] = json.dumps({
            'env_name':'AIWorkerPourNative-v1', 'type':2,
            'env_kwargs':{'control_dt':DT,'joint_names':NAMES,'vessel_variant':'A'}})
        calibration = self.file.create_group('calibration')
        for name, camera in self.cameras.items():
            c = calibration.create_group(name)
            c.create_dataset('intrinsics', data=np.asarray(camera.get_intrinsics_matrix()))
            c.attrs['resolution_wh'] = self.camera_sizes[name]
            c.attrs['prim_path'] = camera.prim_path
            c.attrs['distortion'] = 'none, pinhole'
        self.file.flush()

    def update_camera_poses(self):
        # Root camera follows the measured wrist transform, avoiding nested Fabric camera transforms.
        parent = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(ROOT+'/right_arm/arm_r_link7'))
        world = self.wrist_mount * parent
        UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/TrainingWristCamera')).GetOrderedXformOps()[0].Set(world)

    def create(self, path, shape, dtype):
        options = dict(shape=(0,)+shape, maxshape=(None,)+shape, chunks=(1,)+shape, dtype=dtype)
        if shape and dtype is not str:
            options['compression'] = 'lzf'
        self.datasets[path] = self.demo.create_dataset(path, **options)

    def capture(self, t):
        # Extra render passes finish GPU readback at this fixed physics state.
        for attempt in range(10):
            frames = {name:camera.get_current_frame() for name,camera in self.cameras.items()}
            ready = all(frame.get('rgb') is not None and np.asarray(frame['rgb']).shape[:2]==self.camera_sizes[name]
                        and camera_time_is_current(float(frame.get('rendering_time', -1)),t)
                        and (self.last is None or float(frame['rendering_time'])>self.last[name+'_time'])
                        for name,frame in frames.items())
            if ready:
                break
            self.sim.render()
            self.capture_retries += 1
        else:
            raise RuntimeError('Training camera readback is stale at '+str(t)+': '+
                               str({k:f.get('rendering_time') for k,f in frames.items()}))
        observation = {
            'joint_pos': np.radians([s.Get() for s in self.robot.states]).astype(np.float32),
            'joint_vel': np.radians([s.GetPrim().GetAttribute('state:angular:physics:velocity').Get()
                                    for s in self.robot.states]).astype(np.float32),
        }
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for name,frame in frames.items():
            rgb = np.asarray(frame['rgb'])[:,:,:3].copy()
            reference = frame['rendering_frame']
            observation[name+'_image'] = rgb
            observation[name+'_time'] = float(frame['rendering_time'])
            observation[name+'_reference'] = np.array([
                reference['referenceTimeNumerator'], reference['referenceTimeDenominator']], dtype=np.int64)
            observation[name+'_world_T_camera'] = np.array(
                cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.cameras[name].prim_path))).T
            self.max_skew = max(self.max_skew, abs(observation[name+'_time']-t))
            encoded = io.BytesIO()
            Image.fromarray(rgb).save(encoded, format='JPEG', quality=95)
            self.preview.cameras[name] = encoded.getvalue()
            observation[name+'_encoded'] = np.frombuffer(encoded.getvalue(),dtype=np.uint8).copy()
        return observation

    def append(self, start, action, action_time, row, teacher_action=None):
        now = self.capture(row['t'])
        if self.count and start <= float(self.demo['time'][-1]):
            raise RuntimeError('Non-monotonic observation time')
        if abs(row['t']-start-DT)>1e-6:
            raise RuntimeError('Control transition does not span 0.05 seconds')
        values = {'time':start, 'next_time':row['t'], 'action_time':action_time,
                  'actions':np.radians(action).astype(np.float32), 'rewards':0., 'dones':0,
                  'ground_truth/bottle_pose':row['bottle_pose_xyz']+row['bottle_quat_xyzw']}
        if 'expert_supervision' in self.datasets:
            values['expert_supervision']=int(getattr(self.robot,'expert_active',False))
        if 'teacher_actions' in self.datasets:
            if teacher_action is None: raise RuntimeError('Missing clean expert supervision')
            values['teacher_actions']=np.radians(teacher_action).astype(np.float32)
        values.update({'obs/'+k:v for k,v in self.last.items()})
        values.update({'next_obs/'+k:v for k,v in now.items()})
        values.update({'ground_truth/'+k:row[k] for k in (
            'cup_ml','bottle_ml','spilled_ml','airborne_ml','cup_inventory_fraction',
            'bottle_tilt_deg','command_angle_deg','grasp_verified','stop_command')})
        values['ground_truth/stop_reason'] = row['stop_reason'] or ''
        values['ground_truth/phase'] = row['phase']
        if self.image_encoding=='jpeg95':
            for name in self.cameras:
                values['obs/'+name+'_image']=self.last[name+'_encoded']
                values['next_obs/'+name+'_image']=now[name+'_encoded']
        for key, dataset in self.datasets.items():
            dataset.resize(self.count+1, axis=0)
            dataset[self.count] = values[key]
        self.count += 1
        self.last = now
        if self.count%20==0:
            self.file.flush()
        for label, condition in (
            ('initial',self.count==1),('approach',abs(row['t']-.8)<DT/2),('grasp',abs(row['t']-3.)<DT/2),
            ('pour',abs(row['t']-6.9)<DT/2),('stop',row['stop_command'] and not (self.folder/'stop-task.png').exists())):
            if condition:
                for name in self.cameras:
                    Image.fromarray(now[name+'_image']).save(self.folder/(label+'-'+name+'.png'))

    def finish(self, metrics):
        if self.closed:
            return
        self.demo.attrs['num_samples'] = self.count
        self.file['data'].attrs['total'] = self.count
        self.demo['dones'][-1] = 1
        self.demo['rewards'][-1] = float(metrics['eligible'])
        for name in self.cameras:
            Image.fromarray(self.last[name+'_image']).save(self.folder/('final-'+name+'.png'))
        self.file.attrs['physical_success'] = bool(metrics.get('surface_task_success_zero_spill',metrics['eligible']))
        self.file.attrs['full_task_success']=bool(metrics['eligible'] and not metrics.get('task_scope')=='approach_lift_prefix')
        self.file.attrs['capture_retries'] = self.capture_retries
        self.file.attrs['max_camera_state_skew_s'] = self.max_skew
        self.file.attrs['metrics'] = json.dumps(metrics)
        self.file.attrs['status'] = 'complete'
        self.file.flush()
        self.file.close()
        self.closed = True
        final = self.folder/'episode.hdf5'
        self.path.replace(final)
        for camera in self.cameras.values():
            camera.pause()
        self.cameras.clear()
        return final

    def abort(self, reason):
        if not self.closed and hasattr(self,'file'):
            self.file.attrs['status'] = 'incomplete'
            self.file.attrs['error'] = reason
            self.file.flush()
            self.file.close()
            self.closed = True
        for camera in self.cameras.values():
            camera.pause()
        self.cameras.clear()
