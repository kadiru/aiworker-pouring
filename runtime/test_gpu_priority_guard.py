import json
import unittest
from unittest.mock import patch
import gpu_priority_guard as guard

CID = "a" * 64
OWN = json.dumps([{"Id": CID, "Name": "/aiworker_pouring_4070",
                   "Config": {"Labels": {"owner": "aiworker-pouring"}}}])

class GuardTests(unittest.TestCase):
    def test_never_stops_unowned_container(self):
        with patch.object(guard, "output", return_value=OWN.replace("aiworker-pouring", "unrelated")), \
             patch.object(guard.subprocess, "run") as stop:
            with self.assertRaises(SystemExit): guard.main(CID)
            stop.assert_not_called()

    def test_own_gpu_process_and_other_gpu_do_not_trigger_stop(self):
        rows = guard.GPU + ", 41\nGPU-other, 99"
        with patch.object(guard, "output", side_effect=[OWN, "true", rows, "PID\n41", "false"]), \
             patch.object(guard.time, "sleep"), patch.object(guard, "log"), \
             patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_not_called()

    def test_foreign_4070_process_stops_only_owned_id(self):
        rows = guard.GPU + ", 41\n" + guard.GPU + ", 99"
        with patch.object(guard, "output", side_effect=[OWN, "true", rows, "PID\n41"]), \
             patch.object(guard.Path, "read_text", side_effect=["99 (python) R 1", "0::/user.slice/foreign"]), \
             patch.object(guard, "log"), patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_called_once_with(["docker", "stop", "--timeout", "5", CID], check=True, timeout=20)

    def test_stale_gpu_record_after_process_exit_does_not_trigger_stop(self):
        rows = guard.GPU + ", 99"
        with patch.object(guard, "output", side_effect=[OWN, "true", rows, "PID\n41", "false"]), \
             patch.object(guard.Path, "read_text", side_effect=FileNotFoundError), \
             patch.object(guard.time, "sleep"), patch.object(guard, "log"), \
             patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_not_called()

    def test_owned_child_missing_from_top_is_recognized_by_full_cgroup_id(self):
        rows = guard.GPU + ", 99"
        with patch.object(guard, "output", side_effect=[OWN, "true", rows, "PID\n41", "false"]), \
             patch.object(guard.Path, "read_text", side_effect=["99 (python child) R 1", "0::/system.slice/docker-"+CID+".scope"]), \
             patch.object(guard.time, "sleep"), patch.object(guard, "log"), \
             patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_not_called()

    def test_zombie_gpu_record_does_not_trigger_stop(self):
        rows = guard.GPU + ", 99"
        with patch.object(guard, "output", side_effect=[OWN, "true", rows, "PID\n41", "false"]), \
             patch.object(guard.Path, "read_text", return_value="99 (python) Z 1"), \
             patch.object(guard.time, "sleep"), patch.object(guard, "log"), \
             patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_not_called()

    def test_unreadable_process_identity_still_yields_after_three_failures(self):
        rows = guard.GPU + ", 99"
        with patch.object(guard, "output", side_effect=[OWN]+["true",rows,"PID\n41"]*3), \
             patch.object(guard.Path, "read_text", side_effect=PermissionError), \
             patch.object(guard.time, "sleep"), patch.object(guard, "log"), \
             patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_called_once_with(["docker", "stop", "--timeout", "5", CID], check=True, timeout=20)

    def test_removed_container_does_not_target_name_replacement(self):
        error = guard.subprocess.CalledProcessError(1, ["docker", "inspect", CID])
        with patch.object(guard, "output", side_effect=[OWN, error, "b" * 64]), \
             patch.object(guard, "log"), patch.object(guard.subprocess, "run") as stop:
            guard.main(CID)
            stop.assert_not_called()

if __name__ == "__main__":
    unittest.main()
