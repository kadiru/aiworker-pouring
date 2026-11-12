"""Analytic visual-shell mass proposal, never a runtime physics update.

Integrates the 20-sided visual shell and circular base. The collider sectors
overlap slightly and PhysX's inferred inertia may differ; native mass-property
and pickup checks remain required before using this proposal.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"runtime/src"))
from pour_bottle_geometry import moments


def main():
    project=Path(__file__).resolve().parents[1];root=project/'evidence/scale-A'
    source=root/'bottle-geometry-inventory.json';inventory=json.loads(source.read_text())
    baseline=next(s for s in inventory['candidates'] if s['height_scale']==s['body_diameter_scale']==1.)
    density=.18/moments(baseline['profile_m'])['volume_m3'];rows=[]
    for s in inventory['candidates']:
        rows.append(dict(height_scale=s['height_scale'],body_diameter_scale=s['body_diameter_scale'],
            profile_m=s['profile_m'],**moments(s['profile_m'],density=density)))
    report=dict(cases=rows,calibrated_effective_density_kg_m3=density,baseline_authored_mass_kg=.18,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_modified=False,physically_verified=False,
        scope='Analytic proposal from the authored visual shell with fixed wall/base thickness and mouth. Effective density is chosen to reproduce the existing 180 g baseline; it is not a measured glass density. Collider overlap and native inferred inertia need separate comparison. No fluid mass is added to the rigid bottle; particles remain separate native bodies. No grasp or autonomy claim.')
    out=root/'bottle-mass-proposal.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(cases=len(rows),mass_range_g=[min(r['mass_kg'] for r in rows)*1000,max(r['mass_kg'] for r in rows)*1000],runtime_modified=False)))


if __name__=='__main__':main()
