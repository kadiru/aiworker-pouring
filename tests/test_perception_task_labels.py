import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/src'),str(ROOT/'tools')]
from perception_task_labels import task_level_frames

class TaskLabels(unittest.TestCase):
    def data(self):
        rows=[dict(t=(i+1)*.05,cup_fill_fraction=.91,cup_ml=60.,ever_table_ml=0.,spilled_ml=0.,airborne_ml=0.) for i in range(40)]
        decisions=[dict(t=i*.05,cup_xy=[.5,-.2],visual_grasp_confirmed=True,fill_mode='settle',pause_count=2,stop_latched=None) for i in range(40)]
        return rows,decisions
    def test_future_transfer_does_not_change_earlier_observation_label(self):
        rows,decisions=self.data();base=task_level_frames(rows,decisions)
        for row in rows[20:]:row.update(cup_fill_fraction=.98,cup_ml=65.)
        changed=task_level_frames(rows,decisions)
        self.assertEqual([r for r in base if r['index']<=20],[r for r in changed if r['index']<=20])
        self.assertTrue(all(r['target']==.91 for r in changed if r['index']<=20))
    def test_rejects_surface_jump_and_lost_fluid(self):
        rows,decisions=self.data();rows[19]['cup_fill_fraction']=.87
        self.assertNotIn(20,[r['index'] for r in task_level_frames(rows,decisions)])
        rows,decisions=self.data();rows[19]['ever_table_ml']=.005832
        self.assertTrue(all(r['index']<20 for r in task_level_frames(rows,decisions)))
    def test_requires_observed_grasp_and_stopped_flow(self):
        rows,decisions=self.data()
        for d in decisions:d['visual_grasp_confirmed']=False
        self.assertEqual(task_level_frames(rows,decisions),[])
        rows,decisions=self.data()
        for d in decisions:d['fill_mode']='coarse'
        self.assertEqual(task_level_frames(rows,decisions),[])
    def test_rejects_misaligned_action_and_image_time(self):
        rows,decisions=self.data();decisions[12]['t']+=.05
        with self.assertRaises(AssertionError):task_level_frames(rows,decisions)

    def test_post_stop_return_is_excluded_even_with_stable_height(self):
        rows,decisions=self.data()
        for d in decisions[20:]:d['stop_latched']='cup_near_full'
        frames=task_level_frames(rows,decisions)
        self.assertTrue(frames)
        self.assertTrue(all(f['index']<20 for f in frames))
        # An empty stop has no level decision to supervise on the way home.
        for d in decisions:d.update(fill_mode='coarse',stop_latched='bottle_empty')
        self.assertEqual(task_level_frames(rows,decisions),[])

if __name__=='__main__':unittest.main()
