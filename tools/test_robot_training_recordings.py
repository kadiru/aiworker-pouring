"""Check migration inventory and bind-mount rules without touching real data."""
import tempfile
from pathlib import Path
import unittest
import configure_robot_training_mount as mount
import activate_robot_training_recordings as activate
from stage_robot_training_recordings import digest, inventory, SOURCE, DEST


class RecordingMigrationTests(unittest.TestCase):
    def test_requires_primary_mount_configuration(self):
        with self.assertRaises(RuntimeError):
            activate.proposed_fstab('# no new partition\n')

    def test_preserves_config_and_idempotent(self):
        original = '# root\nUUID=old / ext4 defaults 0 1\n' + mount.ENTRY + '\n'
        result = activate.proposed_fstab(original)
        self.assertTrue(result.startswith(original))
        self.assertEqual(activate.proposed_fstab(result), result)

    def test_refuses_conflicting_target(self):
        with self.assertRaises(RuntimeError):
            activate.proposed_fstab(mount.ENTRY + f'\n/other {SOURCE} none bind 0 0\n')

    def test_refuses_conflicting_source(self):
        with self.assertRaises(RuntimeError):
            activate.proposed_fstab(mount.ENTRY + f'\n{DEST} /other none bind 0 0\n')

    def test_refuses_duplicate_bind_entries(self):
        with self.assertRaises(RuntimeError):
            activate.proposed_fstab(mount.ENTRY + '\n' + activate.ENTRY + '\n' + activate.ENTRY + '\n')

    def test_inventory_notices_extra_files_and_does_not_follow_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'file').write_bytes(b'data')
            (root / 'link').symlink_to('/does/not/exist')
            a = inventory(root)
            self.assertEqual(a['link']['kind'], 'link')
            self.assertEqual(a['file']['size'], 4)
            (root / 'extra').write_bytes(b'x')
            self.assertNotEqual(digest(a), digest(inventory(root)))


if __name__ == '__main__':
    unittest.main()
