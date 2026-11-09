import copy
import unittest
from collections import defaultdict
from scale_data_admission import SIZES,admit,balanced_frame_weights


def fixture():
    plan=dict(collect=[],holdout=[]);audit=dict(records=[],complete=True,plan_sha256='synthetic-plan',checkpoint_sha256='synthetic-weights')
    for i,(h,d) in enumerate(SIZES):
        for split,levels in (('train',(.2,.5,.8,.93)),('validation',(.25,.92)),('holdout',(.94,))):
            for j,target in enumerate(levels):
                name=f'synthetic_{i}_{split}_{j}'
                s=dict(name=name,study_split=split,cup_height_scale=h,cup_diameter_scale=d,
                       calibration_only=True,policy_evaluation=False)
                if split=='holdout':plan['holdout'].append(s);continue
                plan['collect'].append(s)
                audit['records'].append(dict(name=name,split=split,path='/synthetic/'+name,sha256='synthetic-'+name,
                    label=dict(accepted=True,target=target),recording_validation=dict(valid=True)))
    return plan,dict(seed_and_camera_accepted=True,plan_sha256='synthetic-plan',checkpoint_sha256='synthetic-weights'),audit


class AdmissionTests(unittest.TestCase):
    def test_all_sizes_and_whole_episodes(self):
        p,m,a=fixture();self.assertEqual(len(admit(p,m,a)),54)

    def test_rejects_holdout_and_alias_leakage(self):
        for mode in ('holdout','alias'):
            p,m,a=fixture()
            if mode=='holdout':a['records'].append(dict(name=p['holdout'][0]['name']))
            else:a['records'][-1]['sha256']=a['records'][0]['sha256']
            with self.assertRaises(ValueError):admit(p,m,a)

    def test_rejects_missing_high_fill_or_failed_physics(self):
        for change in ('coverage','physics','review'):
            p,m,a=fixture()
            if change=='coverage':a['records'][3]['label']['target']=.85
            elif change=='physics':a['records'][0]['label']['accepted']=False
            else:m['seed_and_camera_accepted']=False
            with self.assertRaises(ValueError):admit(p,m,a)

    def test_rejects_stale_audit_and_changed_split(self):
        for mode in ('plan','checkpoint','split'):
            p,m,a=fixture()
            if mode=='split':a['records'][0]['split']='validation'
            else:a[mode+'_sha256']='changed'
            with self.assertRaises(ValueError):admit(p,m,a)

    def test_size_and_episode_probabilities_do_not_favor_baseline(self):
        meta=[]
        for i,(h,d) in enumerate(SIZES):
            for ep in range(i+1):
                meta.extend([dict(name=f'{i}_{ep}',cup_height_scale=h,cup_diameter_scale=d)]*(ep+1))
        weights=balanced_frame_weights(meta);groups=defaultdict(float);episodes=defaultdict(float)
        for m,w in zip(meta,weights):
            groups[(m['cup_height_scale'],m['cup_diameter_scale'])]+=w;episodes[m['name']]+=w
        self.assertAlmostEqual(sum(weights),1.)
        for p in groups.values():self.assertAlmostEqual(p,1/9)
        for i in range(9):
            for ep in range(i+1):self.assertAlmostEqual(episodes[f'{i}_{ep}'],1/(9*(i+1)))


if __name__=='__main__':unittest.main()
