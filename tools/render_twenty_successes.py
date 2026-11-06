"""Render 20 distinct, verified W30 successes from local native recordings.

Success-only editorial selection, not a new benchmark. No network, simulation,
inference or fitting. Preserve the source files and earlier video exports.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

from render_labeled_benchmark import BG, MUTED, GREEN, text, sha, read

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence/workspace-A'
DATA = ROOT / 'data/material-A-workspace'
OUT = E / 'twenty-successful-pours.mp4'
PREVIEW = E / 'twenty-successes-review'
FPS, SIZE = 20, (1280, 720)
FFMPEG, FFPROBE = '/opt/homebrew/bin/ffmpeg', '/opt/homebrew/bin/ffprobe'
sys.path.insert(0, str(ROOT / 'runtime/src'))
from pour_robust_report import completed_row
from validate_pour_recording import validate


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def strict_success(row):
    goal = (row['cup_surface_height_fraction'] >= .9
            if row['expected_stop'] == 'cup_near_full' else
            row['source_remaining_ml'] <= max(.25, .01 * row['source_initial_ml']))
    return bool(row['grasp_verified'] and row['visual_grasp_confirmed'] and
                row['sustained_grasp_fraction'] >= .8 and row['delivered_to_cup_ml'] >= 1 and
                row['policy_completed'] and row['completed_return_release_sequence'] and
                row['stop_reason'] == row['expected_stop'] and not row['gates'] and goal and
                row['final_outside_ml'] == 0 and row['ever_below_worktop_ml'] == 0 and
                not row['teacher_fallback'])


def choose(report):
    successes = sorted((r for r in report['cases'] if strict_success(r)), key=lambda r: r['name'])
    assert len(successes) == 23
    full = [r for r in successes if r['expected_stop'] == 'cup_near_full']
    empty = [r for r in successes if r['expected_stop'] == 'bottle_empty']
    assert (len(full), len(empty)) == (14, 9)
    # Select by placement coverage only, never by appearance or a relaxed score.
    vectors = {r['name']: np.array([
        (r['cup_world'][0] - .49) / .024, (r['cup_world'][1] + .15) / .024,
        (r['source_xy'][0] + .12) / .027, r['source_xy'][1] / .012]) for r in full}
    selected = [full[0]]
    while len(selected) < 11:
        remaining = [r for r in full if r not in selected]
        selected.append(max(remaining, key=lambda r: min(
            float(np.linalg.norm(vectors[r['name']] - vectors[s['name']])) for s in selected)))
    selected.sort(key=lambda r: r['name'])
    ordered = [item for pair in zip(selected[:9], empty) for item in pair] + selected[9:]
    assert len(ordered) == len({r['name'] for r in ordered}) == 20
    return ordered, [r['name'] for r in successes if r not in ordered]


def audit_sources(report, rows):
    integrity = read(E / 'recording-integrity.json')
    indexed = {r['name']: r for r in integrity['files']}
    checked, validations = [], []
    all_paths = []
    for row in rows:
        native = indexed[row['name']]
        folder = str(Path(native['relative_path']).parent)
        records = [native] + [s for s in integrity['sidecar_files']
                              if str(Path(s['relative_path']).parent) == folder]
        assert len(records) == 5
        for record in records:
            path = DATA / record['relative_path']
            assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], path
            checked.append({k: record[k] for k in ('relative_path', 'bytes', 'sha256')})
        path = DATA / native['relative_path']
        rebuilt = completed_row(path.parent, report['checkpoint_sha256'])
        for key, value in rebuilt.items():
            if key != 'path':
                assert row[key] == value, (row['name'], key)
        assert strict_success(rebuilt) and rebuilt['surface_goal_zero_spill_success']
        result = validate(path)
        assert result['valid'] and result['physical_success']
        validations.append({k: result[k] for k in ('valid', 'transitions', 'control_hz',
                           'duration_sim_s', 'max_camera_state_skew_s', 'physical_success')})
        with h5py.File(path, 'r') as f:
            assert f.attrs['evidence_kind'] == 'learned_visual_policy'
            assert f.attrs['image_encoding'] == 'jpeg95'
            embedded = json.loads(f.attrs['metrics'])
            sidecar = read(path.parent / 'metrics.json')
            # These two metadata fields are added after the HDF5 closes.
            assert sidecar.keys() - embedded.keys() == {'recording', 'wall_seconds'}
            assert all(sidecar[key] == value for key, value in embedded.items())
        all_paths.append(path)
        print('Verified ' + row['name'], flush=True)
    assert len({sha(p) for p in all_paths}) == 20
    return all_paths, dict(native_recordings=20, native_sidecars=80,
                           files=checked, validations=validations,
                           strict_scores_independently_recomputed=20,
                           all_native_hashes_and_sizes_match=True)


def heading(title, eyebrow):
    im = Image.new('RGB', SIZE, BG)
    d = ImageDraw.Draw(im)
    text(d, (72, 52), eyebrow, 17, MUTED, True)
    text(d, (72, 101), title, 48, bold=True)
    d.line((72, 176, 1208, 176), fill='#344552', width=2)
    return im, d


def intro():
    im, d = heading('20 successful pours', 'AIWORKER  /  ISAAC SIM  /  AUTONOMOUS TEST RECORDINGS')
    text(d, (72, 212), 'TASK', 16, MUTED, True)
    text(d, (72, 249), 'Locate  →  Grasp & lift  →  Pour  →  Stop  →  Set down', 31, bold=True)
    text(d, (72, 321), 'Stop when the cup is nearly full or the bottle is empty.', 27)
    text(d, (72, 395), '11 near-full stops   /   9 empty-bottle stops', 31, GREEN, True)
    text(d, (72, 463), '20 selected successes from an earlier 23/30 benchmark.', 26)
    text(d, (72, 506), 'Small position changes around one calibrated arrangement.', 24, MUTED)
    text(d, (72, 588), 'Native task + wrist cameras  ·  Complete episodes at 2×', 24)
    text(d, (72, 646), 'Material A: lightly frosted bottle and cup, with red liquid.', 20, MUTED)
    return im


def method():
    im, d = heading('How the robot learned', 'METHOD  /  SUPERVISED LEARNING + PROGRAMMED FEEDBACK')
    text(d, (72, 214), 'LEARNED COMPONENTS', 17, GREEN, True)
    text(d, (666, 214), 'CONTROLLER', 17, GREEN, True, max_width=542)
    for y, title, detail in [
        (259, 'Pour motion: behavioral cloning', '35 training / 8 validation demonstrations'),
        (370, 'Settled level: ridge regression', '98 training / 28 validation episodes'),
        (481, '16 level coefficients; λ = 0.001', '3,412 training / 960 validation frames')]:
        text(d, (72, y), title, 25, bold=True, max_width=550)
        text(d, (72, y + 39), detail, 21, MUTED, max_width=550)
    for y, line in [(259, 'RGB localization + inverse kinematics'),
                    (305, 'Grip checks, feedback stop and return'),
                    (370, 'Frozen visual networks estimate liquid'),
                    (416, '2 RGB views + 11 measured joints'),
                    (481, 'Goal: ≥90% rim height OR empty'),
                    (527, 'Zero measured loss; upright release')]:
        text(d, (666, y), line, 23, max_width=542)
    text(d, (72, 606), 'Level fit minimizes weighted squared prediction error + ridge penalty.', 22, MUTED)
    text(d, (72, 654), 'Previously fitted on the RTX 4070. This reel uses the frozen model; no new training.', 21)
    return im


def outro():
    im, d = heading('20 selected successes', 'RESULTS  /  SIMULATION ONLY')
    text(d, (72, 219), '11 near-full cups   /   9 empty bottles', 34, GREEN, True)
    text(d, (72, 289), 'Grasp, pour, correct stop, upright set-down and release verified.', 26)
    text(d, (72, 370), 'Full source benchmark: 23/30 passed (76.7%).', 29, bold=True)
    text(d, (72, 421), 'Later wider / swapped layouts: 1/14 passed.', 26)
    text(d, (72, 472), 'This success-only reel does not measure performance on wider layouts.', 23, MUTED)
    text(d, (72, 555), 'Zero measured loss: none outside at finish; none sampled near the table.', 23)
    text(d, (72, 598), 'Measurements at 20 Hz. Transient particles can return to the vessels.', 22, MUTED)
    text(d, (72, 654), 'Existing recordings assembled on the Mac CPU. No Ubuntu jobs changed.', 22)
    return im


PHASES = {
    'Waiting for consistent localization': 'LOCATE BOTTLE + CUP',
    'Visual grasp: approach': 'APPROACH BOTTLE',
    'Visual grasp: close': 'CLOSE GRIPPER',
    'Visual grasp: lift': 'LIFT BOTTLE',
    'Visual grasp: verify': 'VERIFY GRASP',
    'Pour control: coarse': 'POUR',
    'Pour control: brake': 'SLOW THE FLOW',
    'Pour control: settle': 'WAIT FOR LIQUID TO SETTLE',
    'Pour control: trim': 'TOP UP',
    'Returning upright': 'STOP + RETURN UPRIGHT',
    'Setting bottle down': 'SET BOTTLE DOWN',
    'Releasing bottle': 'RELEASE BOTTLE',
    'Retreating': 'RETREAT',
    'Holding completed pose': 'COMPLETE',
}


def template(number, row):
    im = Image.new('RGB', SIZE, BG)
    d = ImageDraw.Draw(im)
    native = row['name'].replace('W30_A_test_', 'W30-')
    text(d, (72, 18), f'AIWORKER  /  AUTONOMOUS POURING  /  {native}', 14, MUTED, True)
    goal = 'Stop near full' if row['expected_stop'] == 'cup_near_full' else 'Stop on empty'
    text(d, (72, 48), f'{number:02} / 20   {goal}', 32, bold=True, max_width=870)
    d.rounded_rectangle((1008, 45, 1208, 88), radius=10, fill='#1A3E33')
    text(d, (1108, 66), 'FINAL: PASS', 18, GREEN, True, max_width=190, anchor='mm')
    text(d, (72, 104), 'TASK CAMERA', 13, MUTED, True)
    text(d, (696, 104), 'WRIST CAMERA', 13, MUTED, True)
    result = (f"Final cup: {100 * row['cup_surface_height_fraction']:.1f}% of rim  /  Target ≥90%"
              if row['expected_stop'] == 'cup_near_full' else
              f"Final bottle: {100 * row['source_remaining_ml'] / row['source_initial_ml']:.1f}% remaining  /  Empty stop verified")
    text(d, (72, 691), result + '  /  Zero measured loss', 18, MUTED)
    return im


def compose(base, demo, index, group, phase, t, hold=None):
    im = base.copy()
    for x, camera in ((72, 'task'), (696, 'wrist')):
        data = demo[f'{group}/{camera}_image'][index].tobytes()
        with Image.open(io.BytesIO(data)) as frame:
            assert frame.size == (384, 384)
            im.paste(frame.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS), (x, 127))
    d = ImageDraw.Draw(im)
    text(d, (1208, 104), f'Sim {t:04.1f}s  ·  {hold or "2× playback"}', 13, MUTED,
         anchor='ra', max_width=360)
    text(d, (72, 653), phase, 21, bold=True)
    return im


def main(preview=False):
    report = read(E / 'final-benchmark.json')
    metadata = read(E / 'model-summary.json')
    assert metadata['checkpoint_sha256'] == report['checkpoint_sha256']
    assert len(report['cases']) == 30 and report['counts']['primary_successes'] == 23
    training = metadata['training']
    for key, expected in dict(train_episodes=98, validation_episodes=28, train_frames=3412,
        validation_frames=960, learned_parameters=16, motion_training_episodes=35,
        motion_validation_episodes=8, frozen_visual_encoders=True).items():
        assert training[key] == expected, key
    assert metadata['chosen']['ridge'] == .001
    assert all(not r['name'].startswith(('W30_A_test', 'W30_A_dev'))
               for r in read(E / 'data-manifest.json'))
    wide = read(ROOT / 'evidence/layout-variety-A/final-benchmark.json')['counts']
    assert wide['complete'] == 14 and wide['strict_passes'] == 1
    rows, omitted = choose(report)
    PREVIEW.mkdir(exist_ok=True)
    if preview:
        for name, image in [('intro', intro()), ('method', method()), ('outro', outro())]:
            image.save(PREVIEW / f'preview-{name}.png')
        indexed = {r['name']: r for r in read(E / 'recording-integrity.json')['files']}
        for n, row in enumerate(rows[:2], 1):
            with h5py.File(DATA / indexed[row['name']]['relative_path'], 'r') as f:
                demo = f['data/demo_0']
                k = len(demo['actions']) // 2
                phase = demo['ground_truth/phase'][k].decode().removeprefix('Visual guard: ')
                compose(template(n, row), demo, k, 'next_obs', PHASES[phase],
                        float(demo['next_time'][k])).save(PREVIEW / f'preview-case-{n}.png')
        print(json.dumps(dict(selected=[r['name'] for r in rows], omitted_successes=omitted)))
        return
    assert not OUT.exists(), 'Preserve the completed export'
    paths, audit = audit_sources(report, rows)
    selection = dict(
        selected=[r['name'] for r in rows], omitted_successes=omitted,
        rule='All nine successful empty stops; eleven of fourteen successful full stops. Start at native 00, then greedy farthest-point coverage of four normalized placement coordinates, ties by ascending native ID. Sort full cases by ID, alternate nine full/empty pairs, then the two remaining full cases.',
        success_only=True, full_source_successes=23, full_source_tests=30,
        source_report_sha256=sha(E / 'final-benchmark.json'),
        native_integrity_sha256=sha(E / 'recording-integrity.json'),
        checkpoint_sha256=report['checkpoint_sha256'],
        scene_plan_sha256=report['scene_plan_sha256'],
        later_wider_layout_result=dict(successes=1, tests=14),
        training=metadata['training'], model_summary_sha256=sha(E / 'model-summary.json'),
        audit=audit)
    save_json(E / 'twenty-successes-selection.json', selection)
    partial = OUT.with_suffix('.partial.mp4')
    command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '1280x720', '-framerate', str(FPS),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
               '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    total, timeline, cards, requests = 0, [], [], []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        def emit(im, count=1):
            nonlocal total
            raw = im.tobytes()
            for _ in range(count):
                encoder.stdin.write(raw)
            total += count

        def emit_card(name, im, seconds):
            start = total
            emit(im, seconds * FPS)
            cards.append(dict(name=name, output_start_frame=start, output_end_frame=total))
            requests.append(dict(name=name, stage='card', frame=start + FPS))

        emit_card('task', intro(), 7)
        emit_card('method', method(), 12)
        for number, (row, path) in enumerate(zip(rows, paths), 1):
            base, start, source_frames = template(number, row), total, []
            samples = read(path.parent / 'samples.json')
            with h5py.File(path, 'r') as f:
                demo = f['data/demo_0']
                times = demo['next_time'][:]
                assert len(samples) == len(times)
                assert np.allclose([s['t'] for s in samples], times, atol=1e-6, rtol=0)
                initial_t = float(demo['time'][0])
                initial = compose(base, demo, 0, 'obs', 'STARTING LAYOUT', initial_t, 'Initial frame')
                emit(initial, FPS // 2)
                source_frames.extend([['obs', 0]] * (FPS // 2))
                requests.append(dict(name=row['name'], stage='initial', frame=start + 2))
                # Complete time interval at exactly 2x; no internal cuts or interpolation.
                playing_start = total
                for k in range(0, len(times), 2):
                    phase = samples[k]['phase'].removeprefix('Visual guard: ')
                    emit(compose(base, demo, k, 'next_obs', PHASES[phase], float(times[k])))
                    source_frames.append(['next_obs', k])
                for label, k in [('pour', int(np.argmax([s['airborne_ml'] for s in samples]))),
                                 ('stop', next(i for i, s in enumerate(samples) if s['stop_command']))]:
                    requests.append(dict(name=row['name'], stage=label, frame=playing_start + k // 2))
                k = len(times) - 1
                final = compose(base, demo, k, 'next_obs', 'COMPLETE  ·  BOTTLE RETURNED + RELEASED',
                                float(times[-1]), 'Final frame')
                requests.append(dict(name=row['name'], stage='final', frame=total + FPS))
                emit(final, 2 * FPS)
                source_frames.extend([['next_obs', k]] * (2 * FPS))
                assert len(source_frames) == total - start
                timeline.append(dict(number=number, name=row['name'], strict_success=True,
                    expected_stop=row['expected_stop'], measured_final_rim_fraction=row['cup_surface_height_fraction'],
                    source_remaining_nominal_ml=row['source_remaining_ml'],
                    final_outside_nominal_ml=row['final_outside_ml'], ever_near_table_nominal_ml=row['ever_below_worktop_ml'],
                    peak_nonairborne_outside_nominal_ml=row['peak_nonairborne_outside_ml'],
                    source_initial_nominal_ml=row['source_initial_ml'], cup_initial_nominal_ml=row['cup_initial_ml'],
                    cup_world=row['cup_world'], source_cup_frame_xy=row['source_xy'],
                    source_relative_path=str(path.relative_to(DATA)), source_sha256=sha(path),
                    output_start_frame=start, output_end_frame=total,
                    output_start_s=start / FPS, output_end_s=total / FPS,
                    playback_rate=2., source_start_s=initial_t, source_end_s=float(times[-1]),
                    initial_hold_s=.5, final_hold_s=2., source_frames=source_frames))
            print(f'Encoded {number:02}/20 {row["name"]}: {total/FPS:.2f}s total', flush=True)
        emit_card('results', outro(), 10)
        encoder.stdin.close()
        assert encoder.wait() == 0
    stream = json.loads(subprocess.check_output([FFPROBE, '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(partial)]))['streams'][0]
    assert stream['width'] == 1280 and stream['height'] == 720
    assert int(stream['nb_read_frames']) == total and stream['r_frame_rate'] == '20/1'
    assert stream['pix_fmt'] == 'yuv420p' and stream['codec_name'] == 'h264'
    partial.replace(OUT)
    manifest = dict(video=str(OUT), sha256=sha(OUT), bytes=OUT.stat().st_size,
        frames=total, fps=FPS, duration_s=total / FPS, resolution=list(SIZE),
        selection_sha256=sha(E / 'twenty-successes-selection.json'),
        checkpoint_sha256=report['checkpoint_sha256'], full_source_successes=23, full_source_tests=30,
        cases=timeline, cards=cards, review_frame_requests=sorted(requests, key=lambda r: r['frame']),
        composition='Twenty distinct native two-camera episodes; entire time interval at labeled 2x, 0.5s initial and 2s final holds. No internal cuts, interpolation or generated simulation images. Silent H.264.',
        outcome_labels='Final complete-episode measurements, not live policy predictions. Zero measured loss permits transient particles that return; sampling is 20Hz, physics 400Hz.',
        scope='Earlier centimeter-scale W30 layouts; success-only selection, not 20/20 overall or twenty successful wider/swapped layouts.',
        rendering='Mac CPU only, two ffmpeg encoder threads. No Ubuntu connection, inference, training or simulator operations.')
    save_json(E / 'twenty-successes-manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('video', 'duration_s', 'sha256', 'bytes', 'frames')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    main(parser.parse_args().preview)
