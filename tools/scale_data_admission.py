"""Whole-episode split checks and equal-size sampling for future perception fits.

This is preparation only. It cannot fit models or admit the current pilot-only
plan, and never reads held-out images.
"""
from collections import Counter, defaultdict
import math

SIZES=tuple((h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1))


def size(spec):
    key=(float(spec['cup_height_scale']),float(spec['cup_diameter_scale']))
    if key not in SIZES:raise ValueError('Undeclared cup size')
    return key


def admit(plan,manifest,audit):
    if not manifest.get('seed_and_camera_accepted'):
        raise ValueError('All declared physical seeds and cameras must be reviewed first')
    if not audit.get('complete') or any(not manifest.get(k) or audit.get(k)!=manifest.get(k) for k in ('plan_sha256','checkpoint_sha256')):
        raise ValueError('Recording audit must match the active declared plan and retained model')
    scenes=plan['collect'];holdout=plan['holdout']
    if not scenes or not holdout:raise ValueError('Declare collection and unused holdout scenes before fitting')
    names=[s['name'] for s in scenes+holdout]
    if len(set(names))!=len(names):raise ValueError('Episode IDs overlap between declared splits')
    indexed={r['name']:r for r in audit['records']}
    if len(indexed)!=len(audit['records']):raise ValueError('Duplicate recording audit entries')
    if {s['name'] for s in holdout}&indexed.keys():raise ValueError('Unused holdout already appears in the fitting audit')
    admitted=[];identity_split={};path_split={};coverage=defaultdict(list)
    for s in scenes:
        split=s['study_split']
        if split not in ('train','validation'):raise ValueError('Only declared train/validation scenes may enter fitting')
        if not s.get('calibration_only') or s.get('policy_evaluation'):
            raise ValueError('Only authored calibration scenes admitted by this helper')
        r=indexed[s['name']];label=r['label'];key=size(s)
        if r['split']!=split:raise ValueError('Recorded split differs from the declaration')
        if not label['accepted'] or not r['recording_validation']['valid']:
            raise ValueError('Rejected physical label or incomplete recording')
        target=float(label['target'])
        if not math.isfinite(target) or not 0<=target<=.98:raise ValueError('Invalid stationary surface label')
        for table,identity in ((identity_split,r['sha256']),(path_split,r['path'])):
            if identity in table:raise ValueError('Recording reused or aliased across episodes')
            table[identity]=split
        coverage[(split,key)].append(target)
        admitted.append((s,r,split))
    for split in ('train','validation'):
        for key in SIZES:
            values=coverage[(split,key)]
            if len(values)<(4 if split=='train' else 2):raise ValueError('Insufficient declared episodes for a cup size')
            if min(values)>.35 or max(values)<.9:raise ValueError('Each size/split needs low and near-full physical coverage')
    if {size(s) for s in holdout}!=set(SIZES):raise ValueError('Unused holdout lacks a declared cup size')
    return admitted


def balanced_frame_weights(meta):
    """Each size gets 1/9 probability; episodes within a size are equally likely."""
    if not meta:raise ValueError('Empty training frames')
    episode_size={};counts=Counter()
    for m in meta:
        key=size(m);name=m['name']
        if name in episode_size and episode_size[name]!=key:raise ValueError('Episode changes cup size')
        episode_size[name]=key;counts[name]+=1
    if set(episode_size.values())!=set(SIZES):raise ValueError('Training frames lack a declared size')
    episodes=Counter(episode_size.values())
    return [1./(len(SIZES)*episodes[size(m)]*counts[m['name']]) for m in meta]
