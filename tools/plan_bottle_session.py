"""Generate an eight-hour maximum session plan; no jobs or scenes are launched."""
import hashlib,json,sys
from pathlib import Path
from bottle_curriculum_geometry import geometry
HERE=Path(__file__).resolve().parents[1]


def main():
    out=HERE/'evidence/bottle-size-A';out.mkdir(exist_ok=True)
    shapes=[dict(id=f'B_h{h:.3f}_w{w:.3f}',**geometry(h,w))
            for w in (1.,.95,.9) for h in (.85,.925,1.,1.075,1.15)]
    withheld=[dict(id=f'BH_h{h:.3f}_w{w:.3f}',**geometry(h,w))
              for h in (.9,1.1) for w in (.925,.975)]
    stages=[
        dict(id='readiness',minutes=30,work='Freeze input/source/checkpoint hashes; verify idle owned container, GPU guard, disk budget, closed recordings and all current level/tracker gates.'),
        dict(id='dry_geometry',minutes=45,max_episodes=45,work='Native derived collision/mass/inertia inspection, IK clearance, then up to three grasp/lift/tilt/side-return/release trials per 15 development shapes; zero liquid.'),
        dict(id='calibration',minutes=150,max_episodes=120,work='Up to 90 training and 30 development-validation episodes, split by complete episode before capture. Six train/two validation per shape. Balance low/intermediate/high cup fills and static/moving source views; vary table layout. Native integrity and actual camera review required.'),
        dict(id='supervised_fit',minutes=75,max_candidates=3,work='Independent RGB/color and bottle localization/source-presence fits, plus contextual BC only from verified motion demonstrations if collected. Checkpoint every ten epochs; unchanged accuracy gates; no heldout fitting.'),
        dict(id='development',minutes=60,max_episodes=12,work='Only after all perception/geometry checks pass: six full-stop and six empty-stop development pours covering height/width extremes, simultaneous upright-and-side-return. Preserve all failures; repair only within stage budget.'),
        dict(id='reserved',minutes=90,max_episodes=30,work='Only after development passes, freeze everything; 30 once-only autonomous tests, 15 full/15 empty, including the four withheld intermediate shapes and new layout/fill combinations. No tuning, retries or online updates.'),
        dict(id='archive',minutes=30,work='Close HDF5, hash reports/models/sources, audit all outcomes, produce labeled all-attempt video and commit/push small evidence. Preserve failed or interrupted attempts as failures/incomplete, never silently retry.')]
    assert sum(s['minutes'] for s in stages)==480
    plan=dict(version=1,status='planned_not_launched_runtime_integration_required',max_wall_seconds=28800,
        device_uuid='GPU-cde23916-c5fa-f18e-a343-127f732cc2f6',owned_container_id='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1',
        resource_limits=dict(cpu_quota_cores=4,cpuset='16-23',cpu_shares=256,memory_gib=16,nice=10),
        stages=stages,development_shapes=shapes,reserved_shapes=withheld,
        curriculum_order=['Height only, nominal width: 0.85/1.0/1.15 first','Remaining height variants','Widths .95 then .90 after measured contact checks'],
        unchanged=['Material A','Liquid/particle settings','Accepted tracker t1','Full/empty criteria','Simultaneous upright and lateral return, robot-right preferred, feasible left fallback'],
        required_runtime_integration=['One bottle profile for visual/collision shell, particle initialization/containment and capacity',
            'Geometry-derived mouth/source projection, grasp frame, lift verification and IK',
            'Scaled authored shell mass and native computed inertia/COM inspection',
            'Full bottle/robot/table/cup collision envelopes and release/set-down verification',
            'Neck-relative source/empty image crops and explicit checkpoint size support'],
        gates=dict(existing_18_moving_tests='Run only after all original development gates pass; distinct from these future 30 bottle-task tests',
            level_per_size_mae=.02,near_full_absolute_error=.035,near_full_overestimate=.025,rgb_color_disagreement=.035,localization_mm=3,
            full_settled_height_min=.90,maximum_sampled_height=.98,maximum_lost_particles=5,
            report_strict_zero_separately=True,empty='Retain existing source-empty threshold and confirmation; no threshold relaxation',
            dry='All shapes must pass contact, lift, clearance, upright placement and ordered release before wet use'),
        deadline_policy=dict(clock='Monotonic, persistent start/deadline; restart/resume never resets the eight-hour budget',
            admission='Do not start a trial unless its conservative maximum time plus close/audit margin fits both stage and overall remaining time',
            checkpoints='After every closed episode and every ten optimizer epochs; save model, optimizer, RNG, split, source hashes and progress cursor',
            overrun='Stop admitting work; request owned trial recovery/closure. Never kill a foreign process or restart after GPU-priority shutdown.',
            failure='A failed required gate stops dependent stages; save reports, use remaining time only for declared development work. No automatic heldout retries.',
            stage_budgets='Caps, not promised counts. Incomplete data/validation blocks later stages. Reserve final 30 minutes for closure.'),
        estimate='Episode counts are upper bounds pending measured wall time and native recording size; no guarantee the entire curriculum fits eight hours.',
        hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'tools/bottle_curriculum_geometry.py']})
    (out/'session-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    report=dict(calculated_shapes=len(shapes)+len(withheld),development_shapes=len(shapes),reserved_shapes=len(withheld),
        lip_height_mm=[min(s['lip_height_m'] for s in shapes)*1000,max(s['lip_height_m'] for s in shapes)*1000],
        outer_body_diameter_mm=[min(s['outer_body_diameter_m'] for s in shapes)*1000,max(s['outer_body_diameter_m'] for s in shapes)*1000],
        proposed_empty_mass_kg=[min(s['proposed_empty_mass_kg'] for s in shapes),max(s['proposed_empty_mass_kg'] for s in shapes)],
        all_no_wider_than_baseline=all(s['outer_body_diameter_m']<=.0548+1e-12 for s in shapes+withheld),
        grasp_height_fixed=all(s['grasp_height_m']==.048 for s in shapes+withheld),mouth_fixed=all(s['inner_mouth_diameter_m']==.018 for s in shapes+withheld),
        scope='CPU calculations only; no native shape spawned, no actual grasp/contact or inertia acceptance.',plan_sha256=hashlib.sha256((out/'session-plan.json').read_bytes()).hexdigest())
    (out/'geometry-proposal.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
