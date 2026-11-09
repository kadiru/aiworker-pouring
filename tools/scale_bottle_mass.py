"""Analytic visual-shell mass proposal, never a runtime physics update.

Integrates the 20-sided visual shell and circular base. The collider sectors
overlap slightly and PhysX's inferred inertia may differ; native mass-property
and pickup checks remain required before using this proposal.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def moments(profile,thickness=.0024,sides=20,density=1.):
    p=np.asarray(profile,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Invalid profile')
    if not (p>0).all() or not (np.diff(p[:,0])>0).all() or thickness<=0 or density<=0 or sides<3:
        raise ValueError('Invalid shell dimensions or density')
    # Circular base, then regular-polygon annular sections. Radius varies
    # linearly along each authored segment; three Gauss nodes integrate all
    # degree-four volume/moment integrands exactly up to floating point.
    z0,r0=p[0];R=r0+thickness
    volume=math.pi*R*R*z0
    first=volume*z0/2;second=volume*z0*z0/3
    polar=volume*R*R/2
    area_factor=sides*math.sin(2*math.pi/sides)/2
    polar_factor=sides*math.sin(2*math.pi/sides)*(2+math.cos(2*math.pi/sides))/12
    nodes,weights=np.polynomial.legendre.leggauss(3)
    for (a,ra),(b,rb) in zip(p,p[1:]):
        z=(a+b)/2+nodes*(b-a)/2;w=weights*(b-a)/2
        r=ra+(rb-ra)*(z-a)/(b-a)
        area=area_factor*((r+thickness)**2-r*r)
        q=polar_factor*((r+thickness)**4-r**4)
        volume+=float(w@area);first+=float(w@(area*z));second+=float(w@(area*z*z));polar+=float(w@q)
    center=first/volume;mass=volume*density
    transverse=density*(polar/2+second-volume*center*center)
    inertia=[transverse,transverse,density*polar]
    assert all(v>0 for v in inertia) and 0<center<p[-1,0]
    return dict(volume_m3=volume,mass_kg=mass,center_of_mass_m=[0.,0.,center],diagonal_inertia_kg_m2=inertia)


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
