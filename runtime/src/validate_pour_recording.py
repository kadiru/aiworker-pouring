"""Validate a native pouring episode before admitting it to a training dataset."""
import argparse
import io
from PIL import Image
import json
import sys
from pathlib import Path

import numpy as np
sys.path.append('/workspace/pydeps51')
import h5py
from pour_camera_clock import recording_tolerance


def decode(frame,codec):
    return np.asarray(Image.open(io.BytesIO(frame.tobytes())).convert('RGB')) if codec=='jpeg95' else np.asarray(frame)


def validate(path):
    with h5py.File(path, 'r') as file:
        assert file.attrs['status'] == 'complete', 'Episode is incomplete'
        demo = file['data/demo_0']
        count = int(demo.attrs['num_samples'])
        assert count > 20, 'Episode is too short'
        dt = float(file.attrs['control_dt'])
        starts = demo['time'][:]
        ends = demo['next_time'][:]
        assert np.allclose(ends-starts, dt, atol=1e-6, rtol=0), 'Uneven transitions'
        assert np.allclose(starts[1:], ends[:-1], atol=1e-6, rtol=0), 'Missing transitions'
        assert np.array_equal(starts, demo['action_time'][:]), 'Actions do not start at their observation time'
        assert demo['actions'].shape == (count,11), 'Wrong action dimensions'
        if 'teacher_actions' in demo:
            assert demo['teacher_actions'].shape==(count,11) and np.isfinite(demo['teacher_actions'][:]).all()
            assert file.attrs['evidence_kind']=='noisy_expert_reference'
        assert np.isfinite(demo['actions'][:]).all(), 'Non-finite action'
        assert np.max(abs(demo['actions'][:])) <= np.pi+.01, 'Actions are not in radians'
        assert len(json.loads(file.attrs['joint_names'])) == 11, 'Missing joint order'
        assert file.attrs['variant'] == 'A', 'Unexpected vessel material'
        assert 'ground_truth' not in demo['obs'] and 'bottle_pose' not in demo['obs'], 'Privileged label entered policy inputs'
        codec=file.attrs.get('image_encoding','raw')
        cameras = {}
        max_skew = 0.
        camera_tolerance = recording_tolerance(file.attrs)
        for name in ('task','wrist') + (('cup_detail',) if 'cup_detail' in file['calibration'] else ()):
            size=tuple(int(v) for v in file['calibration/'+name].attrs.get('resolution_wh',(384,384)))
            assert size in ((768,768),(1024,1024)) if name=='cup_detail' else size==(384,384)
            first, last = [], []
            for group, timestamps in [('obs',starts),('next_obs',ends)]:
                image = demo[group+'/'+name+'_image']
                if codec=='jpeg95':
                    assert image.shape==(count,), 'Malformed JPEG frame dataset'
                else:
                    assert image.shape == (count,size[1],size[0],3), 'Missing or malformed camera images'
                    assert image.dtype == np.uint8, 'Images should be raw uint8 RGB'
                camera_time = demo[group+'/'+name+'_time'][:]
                skew = float(np.max(abs(camera_time-timestamps)))
                max_skew = max(max_skew,skew)
                assert skew < camera_tolerance, 'Stale camera frame'
                reference = demo[group+'/'+name+'_reference'][:]
                assert (reference[:,1]>0).all(), 'Invalid render reference'
                assert (np.diff(camera_time)>0).all(), 'Repeated camera timestamp'
                assert np.isfinite(demo[group+'/joint_pos'][:]).all()
                assert np.isfinite(demo[group+'/joint_vel'][:]).all()
                assert demo[group+'/joint_pos'].shape == (count,11)
                assert np.allclose(demo[group+'/'+name+'_world_T_camera'][:,3,:],[0,0,0,1])
                for i in (0,count//2,count-1):
                    assert decode(image[i],codec).shape == (size[1],size[0],3)
                    assert float(decode(image[i],codec).std()) > 5., 'Camera appears blank'
                if group=='obs': first=decode(image[0],codec)
                else: last=decode(image[-1],codec)
            for i in range(1,count):
                assert np.array_equal(demo['obs/'+name+'_image'][i],demo['next_obs/'+name+'_image'][i-1]), 'Image transition continuity failed'
            change=float(np.mean(abs(first.astype(float)-last.astype(float))))
            assert change>.02, 'Camera did not observe any episode motion'
            cameras[name] = dict(resolution=list(size),rgb_dtype='uint8',encoding=codec,
                                 first_last_mean_pixel_change=change,
                                 intrinsics=file['calibration/'+name+'/intrinsics'][:].tolist())
        assert np.allclose(demo['obs/task_time'][:],demo['obs/wrist_time'][:],atol=1e-7,rtol=0)
        assert demo['dones'][-1]==1 and not demo['dones'][:-1].any()
        metrics=json.loads(file.attrs['metrics'])
        prefix=file.attrs.get('task_scope')=='approach_lift_prefix'
        reviewed=file.attrs.get('camera_visual_review')=='passed'
        recovery=file.attrs.get('evidence_kind')=='policy_prefix_expert_recovery'
        recovery_labels=False
        if recovery:
            labels=demo['expert_supervision'][:]
            assert labels.shape==(count,) and np.isin(labels,[0,1]).all()
            indices=np.flatnonzero(labels)
            if len(indices):
                assert not labels[:indices[0]].any() and labels[indices[0]:].all(), 'Expert labels are not a contiguous suffix'
            recovery_labels=len(indices)>20 and indices[0]>0 if len(indices) else False
        return dict(valid=True,path=str(path),transitions=count,control_hz=1/dt,
                    duration_sim_s=float(ends[-1]-starts[0]),max_camera_state_skew_s=max_skew,
                    camera_timestamp_tolerance_s=camera_tolerance,
                    physical_success=bool(file.attrs['physical_success']),
                    training_eligible=not prefix and file.attrs.get('task_scope') not in ('stationary_perception_calibration','moving_perception_calibration') and bool(file.attrs['physical_success']) and reviewed and
                        file.attrs['evidence_kind'] in ('scripted_reference','noisy_expert_reference'),camera=cameras,
                    task_scope=file.attrs.get('task_scope','full_pour'),
                    corrective_training_eligible=bool(prefix and reviewed and metrics.get('prefix_success') and
                        file.attrs['evidence_kind']=='noisy_expert_reference' and 'teacher_actions' in demo),
                    recovery_training_eligible=bool(recovery and reviewed and recovery_labels and
                        metrics.get('recovery_success_zero_spill') and metrics.get('recovery',{}).get('teacher_complete')),
                    metrics=metrics)


def export_views(path):
    from PIL import Image
    with h5py.File(path,'r') as file:
        demo=file['data/demo_0']
        times=demo['next_time'][:]
        stop=np.flatnonzero(demo['ground_truth/stop_command'][:])
        indices={'initial':0,'grasp':int(np.argmin(abs(times-3.))),
                 'pour':int(np.argmin(abs(times-6.9))),'final':len(times)-1}
        if len(stop): indices['stop']=int(stop[0])
        for label,index in indices.items():
            for camera in ('task','wrist'):
                Image.fromarray(decode(demo['next_obs/'+camera+'_image'][index],file.attrs.get('image_encoding','raw'))).save(
                    path.parent/(label+'-'+camera+'.png'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--approve-views', action='store_true',
                        help='Use only after inspecting the saved camera views through grasp and pouring')
    parser.add_argument('--export-views', action='store_true')
    args=parser.parse_args()
    result=validate(args.episode)
    if args.export_views: export_views(args.episode)
    if args.approve_views:
        with h5py.File(args.episode,'r+') as file:
            file.attrs['camera_visual_review']='passed'
        result=validate(args.episode)
    body=json.dumps(result,indent=2,allow_nan=False)
    if args.output: args.output.write_text(body+'\n')
    print(body)
