"""The run index is committed only after native recording sidecars/trajectory."""
import json

def committed_recording_paths(run_folder):
    index=run_folder/'results.json'
    if not index.exists():return set()
    # run_fountain writes this index by atomic rename after samples and NPZ.
    return {row['path'] for row in json.loads(index.read_text())}
