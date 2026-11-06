"""Check the complete encoded reel and compare camera crops with native frames."""
import io
import json
import math
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw

from render_labeled_benchmark import font, sha, read
from render_twenty_successes import E, DATA, OUT, PREVIEW, FFMPEG, FFPROBE, save_json


def main():
    manifest = read(E / 'twenty-successes-manifest.json')
    assert sha(OUT) == manifest['sha256']
    subprocess.run([FFMPEG, '-v', 'error', '-threads', '2', '-i', str(OUT),
                    '-f', 'null', '-'], check=True)
    requests = manifest['review_frame_requests']
    assert len(requests) == 83
    assert len({r['frame'] for r in requests}) == len(requests)
    indices = [r['frame'] for r in requests]
    assert indices == sorted(indices)
    expression = "select='" + '+'.join(f'eq(n,{n})' for n in indices) + "'"
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-threads', '2', '-i', str(OUT),
                    '-vf', expression, '-fps_mode', 'vfr',
                    str(PREVIEW / 'encoded-%03d.png')], check=True)
    cases = {c['name']: c for c in manifest['cases']}
    frames = []
    for number, request in enumerate(requests, 1):
        path = PREVIEW / f'encoded-{number:03}.png'
        assert path.exists()
        frame = dict(**request, path=str(path), sha256=sha(path))
        if request['name'] in cases:
            case = cases[request['name']]
            group, k = case['source_frames'][request['frame'] - case['output_start_frame']]
            comparison = {}
            with h5py.File(DATA / case['source_relative_path'], 'r') as f:
                actual = Image.open(path).convert('RGB')
                for x, camera in ((72, 'task'), (696, 'wrist')):
                    data = f[f'data/demo_0/{group}/{camera}_image'][k].tobytes()
                    native = Image.open(io.BytesIO(data)).convert('RGB').resize(
                        (512, 512), Image.Resampling.LANCZOS)
                    difference = np.asarray(actual.crop((x, 127, x + 512, 639)), dtype=float) - np.asarray(native, dtype=float)
                    mae = float(np.mean(abs(difference)))
                    psnr = float(10 * np.log10(255 ** 2 / max(1e-12, np.mean(difference ** 2))))
                    assert mae < 5, (request, camera, mae)
                    comparison[camera] = dict(mean_absolute_pixel_error=mae, psnr_db=psnr)
            frame.update(source_group=group, source_frame_index=k, native_camera_comparison=comparison)
        frames.append(frame)
    sheets = []
    for page in range(math.ceil(len(frames) / 8)):
        subset = frames[page * 8:(page + 1) * 8]
        sheet = Image.new('RGB', (1280, math.ceil(len(subset) / 2) * 390), '#101820')
        draw = ImageDraw.Draw(sheet)
        for j, frame in enumerate(subset):
            x, y = (j % 2) * 640, (j // 2) * 390
            sheet.paste(Image.open(frame['path']).convert('RGB').resize(
                (640, 360), Image.Resampling.LANCZOS), (x, y))
            label = frame['name'] + ' / ' + ('stop vicinity' if frame['stage'] == 'stop' else frame['stage'])
            draw.text((x + 12, y + 363), label, font=font(16), fill='white')
        path = PREVIEW / f'contact-{page + 1:02}.jpg'
        sheet.save(path, quality=93)
        sheets.append(str(path))
    probe = json.loads(subprocess.check_output([FFPROBE, '-v', 'error', '-show_streams',
                       '-show_format', '-of', 'json', str(OUT)]))
    stream = probe['streams'][0]
    assert len(probe['streams']) == 1 and stream['codec_name'] == 'h264'
    assert (stream['width'], stream['height'], stream['pix_fmt']) == (1280, 720, 'yuv420p')
    assert int(stream['nb_frames']) == manifest['frames']
    assert abs(float(probe['format']['duration']) - manifest['duration_s']) < .001
    report = dict(video_sha256=manifest['sha256'], full_video_decode_passed=True,
        encoded_review_frames=len(frames), distinct_successful_episodes=20,
        all_twenty_have_initial_pour_stop_final=True, native_camera_crops_checked=160,
        native_camera_crops_within_lossy_encoding_tolerance=True,
        max_mean_absolute_pixel_error=max(c['mean_absolute_pixel_error']
            for f in frames for c in f.get('native_camera_comparison', {}).values()),
        duration_s=manifest['duration_s'], codec=stream['codec_name'], pixel_format=stream['pix_fmt'],
        frames=frames, contact_sheets=sheets, visual_review='pending')
    save_json(E / 'twenty-successes-review.json', report)
    print(json.dumps({k: report[k] for k in ('duration_s', 'encoded_review_frames',
        'native_camera_crops_checked', 'max_mean_absolute_pixel_error', 'visual_review')}))


if __name__ == '__main__':
    main()
