"""Measure post-release recontact in closed, checksum-verified recordings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / 'data/material-A-wide'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.output.resolve().is_relative_to(PROJECT)
    assert not args.output.exists()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['native_checks_passed'] and not manifest['pending_attempts']
    sidecars = {x['relative_path']:x for x in manifest['sidecar_files']}
    records = [x for x in manifest['files'] if x['name'].startswith('XW30_A_dev_') and x['name'].endswith(f'_r{args.round}')]
    assert len(records) == 8
    results = []
    for r in records:
        path = DATA/r['relative_path']
        assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256']
        loaded = {}
        for name in ('samples.json','policy-decisions.json'):
            item = path.parent/name
            identity = sidecars[str(item.relative_to(DATA))]
            assert item.stat().st_size == identity['bytes'] and sha(item) == identity['sha256']
            loaded[name] = json.loads(item.read_text())
        samples, decisions = loaded['samples.json'], loaded['policy-decisions.json']
        assert len(samples) == len(decisions)
        indices = [i for i,d in enumerate(decisions) if d['mode'] in ('Retreating','Holding completed pose')]
        contacts = []
        for i in indices:
            found = [c for c in samples[i]['contacts'] if c.startswith('gripper_')]
            if found:
                contacts.append(dict(t=samples[i]['t'],mode=decisions[i]['mode'],contacts=found))
        speed = [float(np.linalg.norm(np.array(samples[i]['bottle_pose_xyz'])-samples[i-1]['bottle_pose_xyz'])/
                       (samples[i]['t']-samples[i-1]['t'])) for i in indices if i]
        results.append(dict(name=r['name'],source_sha256=r['sha256'],retreat_observed=bool(indices),
            sampled_post_release_contact_frames=len(contacts),contacts=contacts,
            maximum_post_release_bottle_speed_m_s=max(speed,default=None),
            clamped_retreat_frames=sum(decisions[i]['delta_clamped'] for i in indices),
            final_settled_rim_fraction=samples[-1]['cup_fill_fraction'],
            final_outside_nominal_ml=samples[-1]['spilled_ml']+samples[-1]['airborne_ml']))
    result=dict(round=args.round,source_manifest_sha256=sha(args.manifest),cases=results,
        all_eight_retreats_observed=all(r['retreat_observed'] for r in results),
        cases_with_post_release_contact=[r['name'] for r in results if r['contacts']],
        scope='20 Hz recorded bottle contact sets, poses and command logs. Diagnostic only; strict task scoring is unchanged. No claim about unsampled contacts or physical hardware.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__ == '__main__':
    main()
