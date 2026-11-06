"""CPU checks for perceptual grasp gating and safe top-up return history."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from pour_guard_actor import Actor,monotone_return_path,topup_goal,trim_expired,slow_empty_evidence
from pour_guard_vision import project,plane_point,bottle_base_from_neck,locate


class GuardTests(unittest.TestCase):
    def test_early_source_loss_returns_slowly_with_grip_closed(self):
        a=object.__new__(Actor);zero=np.zeros(7);middle=np.full(7,.2);current=np.full(7,.4)
        a.profile=dict(grasp_mark_s=0.,lift_mark_s=1.,return_s=.8,place_s=.5,release_s=.3,retreat_s=.5,
            warp=[0.,.2,.7,1.],grip=[1.,0.])
        a.history=[(0.,zero),(1.,zero),(2.,middle),(3.,current)]
        a.outbound=[(0.,zero),(20.,middle),(42.,current)];a.pour_phase=42.;a.qlift=zero
        a.saved=dict(early_return_seconds=4.3);robot=SimpleNamespace()
        a.latch('source_tracking_lost',3.,robot)
        targets,mode=a.terminal.act(3.8)
        self.assertEqual(mode,'Returning upright');self.assertTrue(np.all(targets[:7]>.3))
        for elapsed in np.linspace(.8,4.29,20):
            q,mode=a.terminal.act(3.+elapsed)
            self.assertEqual(mode,'Returning upright');self.assertTrue(np.all((q[:7]>=0)&(q[:7]<=.4)))
            np.testing.assert_allclose(q[7:],np.pi/4)
        self.assertEqual(a.profile['return_s'],.8)

    def test_initial_family_calibration_does_not_change_lift_tracking_scores(self):
        old,new={'family':'retained'},{'family':'outline'}
        bank=dict(alternatives=[old,new],initial_score_maps=[
            dict(offset=.7,scale=.75,anchor=.4),dict(offset=0.,scale=1.,anchor=0.)])
        def match(image,b,*args):
            return (np.array([10.,20.]),.6) if b is old else (np.array([30.,40.]),.8)
        with patch('pour_guard_vision.locate',side_effect=match):
            uv,score=locate(None,bank,device='cpu')
            np.testing.assert_array_equal(uv,[10.,20.]);self.assertAlmostEqual(score,.85)
            uv,score=locate(None,bank,tracking=True,device='cpu',expected_uv=[30.,40.])
            np.testing.assert_array_equal(uv,[30.,40.]);self.assertEqual(score,.8)

    def approach_actor(self):
        a=object.__new__(Actor);a.abort_time=None;a.abort_path=None;a.grasp_stage='approach';a.stage_start=0.
        a.home=np.full(7,.1);a.grasp=np.zeros(7);a.approach_path=np.asarray([a.home,a.grasp])
        a.threshold=dict(approach_tracking_tolerance_rad=.02,approach_tracking_wait_s=.75)
        a.history=[(0.,a.home.copy()),(1.35,np.zeros(7))];a.approach_check=None
        return a

    def test_untracked_approach_keeps_fingers_open(self):
        a=self.approach_actor();q=np.zeros(11);q[0]=np.radians(17.4)
        targets,_=a.grasp_action(None,q,1.5,1.5)
        self.assertEqual(a.grasp_stage,'approach');self.assertIsNone(a.abort_time)
        np.testing.assert_array_equal(targets[7:],0.)
        self.assertFalse(a.approach_check['arrived'])

    def test_reached_approach_can_close(self):
        a=self.approach_actor();q=np.zeros(11)
        a.grasp_action(None,q,1.4,1.4)
        self.assertEqual(a.grasp_stage,'close')
        targets,_=a.grasp_action(None,q,1.6,1.6)
        self.assertTrue((targets[7:]>0).all())

    def test_blocked_approach_aborts_without_grasping_and_reverses_measured_path(self):
        a=self.approach_actor();q=np.zeros(11);q[0]=np.radians(17.4)
        targets,_=a.grasp_action(None,q,2.11,2.11)
        self.assertEqual(a.stop_latched,'approach_unconfirmed')
        np.testing.assert_array_equal(targets[7:],0.)
        np.testing.assert_array_equal(a.abort_path[0],q[:7])
        targets,_=a.grasp_action(None,q,3.2,3.2)
        np.testing.assert_array_equal(targets[:7],a.home)
        np.testing.assert_array_equal(targets[7:],0.)

    def test_constant_visual_estimate_cannot_allow_endless_topup(self):
        self.assertFalse(trim_expired(None,100.,.1))
        self.assertFalse(trim_expired(8.,8.05,.1))
        self.assertTrue(trim_expired(8.,8.1,.1))
        # The total limit also brakes if rate-holding prevents reaching the cap.
        self.assertTrue(trim_expired(7.2,8.1,.9))

    def test_under_target_already_above_old_trigger_can_top_up(self):
        self.assertGreater(topup_goal(.944,.96),.944)
        self.assertLessEqual(topup_goal(.944,.96),.965)

    def test_full_estimate_never_requests_more_liquid(self):
        with self.assertRaises(ValueError):topup_goal(.97,.96)

    def test_calibrated_ray_plane_roundtrip(self):
        cal=dict(T=[[1,0,0,.1],[0,1,0,.2],[0,0,1,2.],[0,0,0,1]],K=[[400,0,192],[0,400,192],[0,0,1]])
        xyz=np.array([.23,.18,1.175]);uv=project(xyz,cal)
        np.testing.assert_allclose(plane_point(uv,xyz[2],cal),xyz,atol=1e-10)

    def test_neck_plane_is_converted_to_base_before_grasp(self):
        cal=dict(T=[[1,0,0,.1],[0,1,0,.2],[0,0,1,2.],[0,0,0,1]],K=[[400,0,192],[0,400,192],[0,0,1]],table_z=1.04)
        neck=np.array([.23,.18,1.175]);base=bottle_base_from_neck(project(neck,cal),cal)
        np.testing.assert_allclose(base,[.23,.18,1.04],atol=1e-10)
        self.assertAlmostEqual(float((base+[0,0,.048])[2]),1.088)

    def test_return_does_not_revisit_topup_peaks(self):
        envelope=[(a,np.full(7,a)) for a in (0,20,40,60,80,100,110)]
        path=monotone_return_path(envelope,80.,np.full(7,80.),np.zeros(7))
        self.assertLessEqual(path.max(),80.)
        self.assertTrue((np.diff(path[:,0])<=0).all())
        np.testing.assert_array_equal(path[-1],np.zeros(7))

    def actor_for_confirmation(self):
        a=object.__new__(Actor);a.abort_time=None;a.grasp_stage='verify';a.stage_start=0.;a.lift_path=np.zeros((2,7))
        a.initial_neck=np.array([150.,135.]);a.expected_lift_neck=np.array([150.,70.]);a.saved={'source_bank':{}}
        a.confirm_votes=0;a.grasp_confirmed=False;a.profile={};a.threshold={'max_attempts':2};a.attempt=1;a.source_score=.9
        a.saved['pour_primitive']={'grasp_closure':{'min_deficit_rad':np.radians(2.)}}
        class FK:
            def fk(self,q):return np.array([0.,0.,.044]),np.eye(3)
        a.fk=FK();a.grasp_tcp=np.zeros(3);a.grasp_check=None;a.physical_votes=0;a.visual_votes=[]
        return a

    def test_visual_flicker_can_accumulate_with_continuous_physical_hold(self):
        a=self.actor_for_confirmation();a.threshold.update(grasp_visual_window=12,verify_seconds=1.25)
        a.visual_votes=[];a.physical_votes=0;q=np.zeros(11);q[7:]=np.radians(38.)
        scores=[.21,.28]*6
        with patch('pour_guard_actor.locate',side_effect=[(np.array([150.,70.]),s) for s in scores]):
            for i in range(11):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
            self.assertFalse(a.grasp_confirmed)
            a.grasp_action(np.zeros((4,4,3)),q,.55,.55)
        self.assertTrue(a.grasp_confirmed)

    def test_physical_contact_loss_clears_old_visual_votes(self):
        a=self.actor_for_confirmation();a.threshold.update(grasp_visual_window=12,verify_seconds=1.25)
        a.visual_votes=[];a.physical_votes=0;q=np.zeros(11);q[7:]=np.radians(38.)
        with patch('pour_guard_actor.locate',return_value=(np.array([150.,70.]),.99)):
            for i in range(5):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
            empty=q.copy();empty[7:]=np.pi/4;a.grasp_action(np.zeros((4,4,3)),empty,.25,.25)
            self.assertEqual(a.physical_votes,0);self.assertEqual(a.visual_votes,[])
            for i in range(5):a.grasp_action(np.zeros((4,4,3)),q,.30+i*.05,.30+i*.05)
            self.assertFalse(a.grasp_confirmed)

    def test_slow_empty_requires_cap_hold_low_flow_and_visible_evidence(self):
        threshold=dict(empty_cap_dwell_seconds=.25,slow_empty_probability=.90,slow_empty_max_flow_ml_s=1.)
        args=dict(threshold=threshold,phase=131.,cap=131.,tilt=130.5,cap_age=1.,held=True,present=.99,empty=.95,flow_ml_s=0.)
        self.assertTrue(slow_empty_evidence(**args))
        for change in (dict(phase=120.),dict(tilt=110.),dict(cap_age=.1),dict(held=False),dict(present=.2),dict(empty=.8),dict(flow_ml_s=2.)):
            self.assertFalse(slow_empty_evidence(**dict(args,**change)),change)

    def test_static_bottle_cannot_enable_pouring(self):
        for window in (0,12):
            with self.subTest(window=window):
                a=self.actor_for_confirmation();a.threshold['grasp_visual_window']=window
                q=np.zeros(11);q[7:]=np.radians(38.)
                with patch('pour_guard_actor.locate',return_value=(np.array([150.,135.]),.99)):
                    for i in range(8):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
                self.assertFalse(a.grasp_confirmed);self.assertEqual(a.confirm_votes,0)

    def test_visual_lift_requires_persistent_confirmation(self):
        a=self.actor_for_confirmation()
        q=np.zeros(11);q[7:]=np.radians(38.)
        with patch('pour_guard_actor.locate',return_value=(np.array([150.,70.]),.99)):
            for i in range(5):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
            self.assertFalse(a.grasp_confirmed)
            a.grasp_action(np.zeros((4,4,3)),q,.25,.25)
        self.assertTrue(a.grasp_confirmed)

    def test_fully_closed_empty_gripper_cannot_confirm(self):
        for window in (0,12):
            with self.subTest(window=window):
                a=self.actor_for_confirmation();a.threshold['grasp_visual_window']=window;q=np.zeros(11);q[7:]=np.pi/4
                with patch('pour_guard_actor.locate',return_value=(np.array([150.,70.]),.99)):
                    for i in range(8):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
                self.assertFalse(a.grasp_confirmed)

    def test_one_finger_obstruction_cannot_confirm(self):
        a=self.actor_for_confirmation();a.threshold['grasp_visual_window']=12;q=np.zeros(11);q[7:]=np.radians(38.);q[9]=np.pi/4
        with patch('pour_guard_actor.locate',return_value=(np.array([150.,70.]),.99)):
            for i in range(8):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
        self.assertFalse(a.grasp_confirmed)

    def test_stalled_arm_cannot_confirm(self):
        class FK:
            def fk(self,q):return np.array([0.,0.,.01]),np.eye(3)
        for window in (0,12):
            with self.subTest(window=window):
                a=self.actor_for_confirmation();a.threshold['grasp_visual_window']=window;q=np.zeros(11);q[7:]=np.radians(38.);a.fk=FK()
                with patch('pour_guard_actor.locate',return_value=(np.array([150.,70.]),.99)):
                    for i in range(8):a.grasp_action(np.zeros((4,4,3)),q,i*.05,i*.05)
                self.assertFalse(a.grasp_confirmed)


if __name__=='__main__':unittest.main()
