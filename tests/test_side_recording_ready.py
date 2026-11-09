import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from side_recording_ready import committed_recording_paths

class RecordingCommitTests(unittest.TestCase):
    def test_metrics_and_partial_index_do_not_close_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'metrics.json').write_text('{}')
            (run/'results.tmp').write_text('[{"path":"/workspace/episode"}]')
            self.assertEqual(committed_recording_paths(run),set())
            (run/'results.tmp').replace(run/'results.json')
            self.assertEqual(committed_recording_paths(run),{'/workspace/episode'})
    def test_existing_index_excludes_next_uncommitted_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'results.json').write_text(json.dumps([{'path':'first'}]))
            (run/'results.tmp').write_text(json.dumps([{'path':'first'},{'path':'second'}]))
            self.assertEqual(committed_recording_paths(run),{'first'})
if __name__=='__main__':unittest.main()
