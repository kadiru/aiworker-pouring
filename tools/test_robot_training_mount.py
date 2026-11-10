"""Check boot-config proposal rules without touching disks or /etc."""
import unittest
from configure_robot_training_mount import ENTRY, UUID, TARGET, proposed_fstab


class MountProposalTests(unittest.TestCase):
    def test_preserves_existing_entries(self):
        original = '# system\nUUID=abc / ext4 defaults 0 1\n'
        new = proposed_fstab(original)
        self.assertTrue(new.startswith(original))
        self.assertTrue(new.endswith(ENTRY + '\n'))

    def test_idempotent(self):
        original = proposed_fstab('# system\n')
        self.assertEqual(proposed_fstab(original), original)

    def test_refuses_same_uuid_elsewhere(self):
        with self.assertRaises(RuntimeError):
            proposed_fstab(f'UUID={UUID} /mnt/elsewhere ext4 defaults 0 2\n')

    def test_refuses_other_disk_at_target(self):
        with self.assertRaises(RuntimeError):
            proposed_fstab(f'UUID=other {TARGET} ext4 defaults 0 2\n')

    def test_refuses_duplicate_entries(self):
        with self.assertRaises(RuntimeError):
            proposed_fstab(ENTRY + '\n' + ENTRY + '\n')

    def test_comments_are_not_live_entries(self):
        self.assertIn('\n' + ENTRY + '\n', proposed_fstab('# ' + ENTRY + '\n'))


if __name__ == '__main__':
    unittest.main()
