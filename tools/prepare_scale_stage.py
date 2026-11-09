"""Native CPU preparation of new physical cup seeds; parent study is immutable."""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/workspace/learning/A-scale-20261109-v1')
PARENT = Path('/workspace/learning/A-side-return-20261108-v1')
sys.path[:0] = ['/workspace/src', '/workspace']
from bottle_pour import SIDES, vessel_mask
from pour_vessel_geometry import cup_profile, cup_geometry
from pour_perception_geometry import calibration_seed_points


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    assert not (ROOT / 'candidate.json').exists()
    frozen = read(PARENT / 'final-selection.json')
    parent = read(PARENT / 'candidate.json')
    assert sha(PARENT / 'best.pt') == frozen['checkpoint_sha256']
    assert all(parent[k] == frozen[k] for k in ('checkpoint_sha256', 'plan_sha256', 'runtime_sources'))
    assert all(sha(Path('/workspace/src') / n) == h for n, h in parent['runtime_sources'].items())
    assert all(sha(Path(p)) == h for p, h in parent['protected_parent_files'].items())
    (ROOT / 'seeds').mkdir()
    shutil.copy2(PARENT / 'best.pt', ROOT / 'best.pt')
    base = read(PARENT / 'plan.json')['development'][0]
    pilots, checks = [], []
    for i, (h, d) in enumerate((h, d) for h in (.9, 1., 1.1) for d in (.9, 1., 1.1)):
        s = copy.deepcopy(base)
        s.update(name=f'SC_A_seed_r0_{i:02d}', study_split='seed_pilot', calibration_only=True,
                 policy_evaluation=False, cup_height_scale=h, cup_diameter_scale=d,
                 calibration_version=4, calibration_initial_height=1., initial_ml=20.,
                 max_seconds=6.5, record_cup_detail=True, record_overview=True,
                 record_surface_diagnostics=True, save_particle_trajectory=True,
                 trajectory_start_s=0., trajectory_stride=1, record_particle_diagnostics=True,
                 cup_detail_resolution=1024, cup_detail_focal_m=.030)
        s.pop('checkpoint', None)
        profile = cup_profile(s)
        spacing = s['spacing']
        assert spacing == .0018
        # Same nominal resolution and physical coefficients. This is only an
        # initial lattice, using the existing calibration minimum pitch ratio.
        pitch = .8 * spacing
        points = []
        for z in np.arange(profile[0][0] + 1.5*spacing, profile[-1][0] - .5*spacing, pitch):
            radius = np.interp(z, [v[0] for v in profile], [v[1] for v in profile]) * np.cos(np.pi/SIDES) - 1.5*spacing
            n = int(radius/pitch)
            points.extend((x*pitch, y*pitch, z) for x in range(-n,n+1) for y in range(-n,n+1)
                          if (x*x+y*y)*pitch*pitch < radius*radius)
        points = np.asarray(points, dtype=np.float32)
        seed = ROOT / 'seeds' / f'cup-{i:02d}-initial.npz'
        np.savez_compressed(seed, cup_local_points=points)
        s.update(calibration_particle_seed=str(seed), calibration_particle_seed_sha256=sha(seed),
                 declared_cup_geometry=cup_geometry(s))
        selected = calibration_seed_points(s, points)
        assert len(selected) == len(points) and vessel_mask(selected, profile).all()
        s['cup_initial_ml'] = len(points)*spacing**3*1e6
        pilots.append(s)
        checks.append(dict(name=s['name'], particle_count=len(points), initial_pitch_m=pitch,
                           nominal_ml=s['cup_initial_ml'], geometry=s['declared_cup_geometry'],
                           seed_sha256=sha(seed), all_initial_centers_inside=True))
    plan = dict(version=1, pilot=pilots, collect=[], holdout=[],
                particle_resolution_m=.0018, practical_lost_particle_limit=5,
                scope='Nine stationary cup-size seed pilots, original bottle and Material A. Authored calibration only, not autonomous pours. Same parent fluid settings and nominal particle resolution.',
                seed_gate=dict(zero_final_and_table_loss=True, stable_final_second=True,
                               high_fill_min_fraction=.85, high_fill_max_fraction=.98,
                               inspect_native_cameras=True),
                component_gates=dict(localization_max_error_mm=3., level_validation_mae=.02,
                                     level_validation_near_max_error=.035, level_validation_near_overestimate=.025),
                future_protocol='Select on every declared cup size with size-balanced sampling and whole-episode train/validation splits. Declare unused static holdouts before fitting. Bottle scaling remains gated pending grasp/physics checks. Report strict zero and <=5 lost-particle task scores; no change to frozen parent scores.')
    protected = dict(parent['protected_parent_files'])
    for p in PARENT.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.pt', '.py'):
            protected[str(p)] = sha(p)
    sources = {p.name: sha(p) for p in Path('/workspace/src').glob('*.py')}
    for p in (ROOT / 'staged').glob('*.py'):
        sources[p.name] = sha(p)
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    manifest = dict(checkpoint_sha256=sha(ROOT/'best.pt'), parent_checkpoint_sha256=frozen['checkpoint_sha256'],
                    plan_sha256=sha(ROOT/'plan.json'), runtime_sources=sources, protected_parent_files=protected,
                    stage='seed_calibration', seed_and_camera_accepted=False, trained=False,
                    validated_cup_scope='baseline_only', method='Retained parent models; no fitting. New native cup-size seed calibration.')
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'seed-preparation.json').write_text(json.dumps(dict(cases=checks,helper_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(dict(pilots=len(pilots),checkpoint_sha256=manifest['checkpoint_sha256'],plan_sha256=manifest['plan_sha256'])))


if __name__ == '__main__':
    main()
