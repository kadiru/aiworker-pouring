"""Make an outcome-labeled reel from verified local Isaac Sim recordings.

This reads closed HDF5 files only. It neither contacts Ubuntu nor runs inference.
The reel deliberately includes two prespecified successes and all seven failures;
it is not a representative-frequency sample of the 30-test benchmark.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence/workspace-A'
DATA = ROOT / 'data/material-A-workspace'
OUT = E / 'labeled-highlights.mp4'
FPS, SIZE = 20, (1280, 720)
BG, WHITE, MUTED = '#101820', '#F3F6F7', '#B3C0CC'
GREEN, RED = '#67D6B1', '#FFAAA3'
FONT = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
BOLD = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
FONTS = {}
STAGES = [0, 1, 7, 16, 17, 22, 24, 26, 27]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def font(size, bold=False):
    key = size, bold
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(str(BOLD if bold else FONT), size)
    return FONTS[key]


def text(draw, xy, value, size, color=WHITE, bold=False, max_width=1120, anchor=None):
    f = font(size, bold)
    assert draw.textlength(value, font=f) <= max_width, value
    draw.text(xy, value, font=f, fill=color, anchor=anchor)


def card(closing=False):
    im = Image.new('RGB', SIZE, BG)
    d = ImageDraw.Draw(im)
    text(d, (80, 54), 'AIWORKER  /  ISAAC SIM', 18, MUTED, True)
    text(d, (80, 109), 'Pouring across wider placements', 42, bold=True)
    d.line((80, 180, 1200, 180), fill='#33414F', width=2)
    text(d, (80, 216), '23 / 30', 84, bold=True)
    text(d, (80, 315), 'complete tasks passed', 28, MUTED)
    text(d, (80, 380), '14 / 15  near-full cup', 25)
    text(d, (80, 422), '9 / 15    empty bottle', 25)
    if not closing:
        text(d, (650, 230), 'What this reel shows', 25, bold=True, max_width=550)
        text(d, (650, 285), '2 success examples', 25, GREEN, max_width=550)
        text(d, (650, 330), 'All 7 failed tests', 25, RED, max_width=550)
        text(d, (650, 395), 'Actual task + wrist camera recordings', 21, MUTED, max_width=550)
        text(d, (650, 433), 'Edited highlights; speed is labeled', 21, MUTED, max_width=550)
        text(d, (80, 579), 'Bottle and cup placement ranges expanded 3×', 25)
        text(d, (80, 635), 'Final outcomes use the complete test, not just the frames shown.', 20, MUTED)
    else:
        text(d, (650, 222), 'The 7 failed tests', 25, bold=True, max_width=550)
        failures = [('2', 'Spill'), ('2', 'Missed empty stop'), ('1', 'Bottle localization rejected'),
                    ('1', 'Cup below target'), ('1', 'Sustained-grip requirement')]
        for n, (count, label) in enumerate(failures):
            y = 274 + n*43
            text(d, (650, y), count, 25, RED, True, max_width=50)
            text(d, (700, y), label, 23, max_width=500)
        text(d, (80, 566), 'Zero spill was not achieved: 9 particles lost in 2 pours.', 26)
        text(d, (80, 618), 'One frozen model. All 30 tests retained. No retries.', 23, MUTED)
        text(d, (80, 663), 'Simulation measurements at 20 Hz • No physical-robot validation', 17, MUTED)
    return im


def labels(i, row):
    height = 100*row['cup_surface_height_fraction']
    pairs = {
        0: ('Stop near full', 'Filled the cup and returned the bottle.',
            f'Final level {height:.1f}%   /   Target ≥90%   /   No measured spill'),
        1: ('Stop on empty', 'Emptied and returned the bottle.',
            '0 particles left in the bottle   /   No measured spill'),
        7: ('Spill', '2 particles ended outside the cup.',
            'Empty stop and return completed   /   Strict zero-spill goal failed'),
        16: ('Localization', 'No pickup: bottle localization was rejected.',
             'Best confidence 0.675   /   Required 0.700   /   No liquid transferred'),
        17: ('Spill', '7 particles ended outside the cup.',
             'Empty stop and return completed   /   Strict zero-spill goal failed'),
        22: ('Early full stop', 'Settled liquid level ended below the target.',
             f'Final level {height:.1f}%   /   Required ≥90%   /   No measured spill'),
        24: ('Missed empty stop', 'Bottle emptied; no stop or return before timeout.',
             'Empty for 24.15 simulated seconds   /   No measured spill'),
        26: ('Missed empty stop', 'Another empty bottle held tilted until timeout.',
             'Empty for 24.60 simulated seconds   /   No measured spill'),
        27: ('Sustained grip', 'Did not meet the sustained-grip requirement.',
             'Both fingers in contact: 76% of tipping samples   /   Required ≥80%')}
    title, reason, detail = pairs[i]
    return f'Test {i+1:02} · {title}', reason, detail


def template(i, row):
    im = Image.new('RGB', SIZE, BG)
    d = ImageDraw.Draw(im)
    title, reason, detail = labels(i, row)
    passed = row['surface_goal_zero_spill_success']
    color = GREEN if passed else RED
    text(d, (80, 18), 'AIWORKER  /  RECORDED BENCHMARK', 14, MUTED, True)
    text(d, (80, 50), title, 31, bold=True, max_width=900)
    d.rounded_rectangle((1030, 47, 1200, 89), radius=10, fill='#1A3E33' if passed else '#40282E')
    text(d, (1115, 67), 'FINAL · PASS' if passed else 'FINAL · FAIL', 17, color, True,
         max_width=158, anchor='mm')
    text(d, (80, 105), 'TASK CAMERA', 13, MUTED, True)
    text(d, (688, 105), 'WRIST CAMERA', 13, MUTED, True)
    text(d, (80, 655), reason, 23, bold=True)
    text(d, (80, 690), detail, 16, MUTED)
    return im


def segments(i, row, times):
    end = float(times[-1]) + .025
    first = float(times[0])
    stop = (row['stop_prediction'] or {}).get('t')
    if i in (0, 1, 22):
        # Continuous full task, followed by a clearly labeled final-frame hold.
        return [(first, end, 2.)]
    if i == 16:
        return [(first, end, 1.)]
    if i in (7, 17):
        return [(first, 4.5, 3.), (9.6 if i == 7 else 10., stop-.65, 2.),
                (stop-.65, stop+.65, .75), (stop+.65, end, 1.5)]
    if i in (24, 26):
        return [(first, 4.5, 3.), (10.3 if i == 24 else 9.7, 16.3, 2.), (16.3, end, 6.)]
    return [(first, 4.5, 3.), (10., stop, 2.), (stop, end, 1.5)]


def compose(base, demo, frame_index, t, rate, *, frozen=False, cut=False):
    im = base.copy()
    for x, camera in ((80, 'task'), (688, 'wrist')):
        encoded = demo['next_obs/'+camera+'_image'][frame_index].tobytes()
        with Image.open(io.BytesIO(encoded)) as frame:
            assert frame.size == (384, 384)
            im.paste(frame.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS), (x, 128))
    d = ImageDraw.Draw(im)
    label = 'Final frame' if frozen else f'{rate:g}× playback'
    if cut:
        label = 'Cut · ' + label
    text(d, (1200, 105), f'Sim {t:04.1f}s  ·  {label}', 13, MUTED,
         max_width=350, anchor='ra')
    return im


def main(preview=False):
    report = read(E/'final-benchmark.json')
    assert report['independently_recomputed_primary_scores'] == 30
    assert report['counts']['primary_successes'] == 23 and report['counts']['zero_spill_runs'] == 28
    rows = {int(x['name'][-2:]): x for x in report['cases']}
    assert [i for i in STAGES if not rows[i]['surface_goal_zero_spill_success']] == [
        i for i,x in rows.items() if not x['surface_goal_zero_spill_success']]
    assert all(rows[i]['surface_goal_zero_spill_success'] for i in STAGES[:2])
    integrity = read(E/'recording-integrity.json')
    indexed = {x['name']: x for x in integrity['files']}
    if preview:
        card().save(E/'labeled-intro-preview.png')
        card(True).save(E/'labeled-outro-preview.png')
        for i in (0, 7, 16, 22, 24, 27):
            row = rows[i]
            with h5py.File(DATA/indexed[row['name']]['relative_path'], 'r') as f:
                demo=f['data/demo_0'];k=len(demo['actions'])-1;t=float(demo['next_time'][k])
                compose(template(i,row),demo,k,t,1.,frozen=True).save(E/f'labeled-preview-{i:02}.png')
        print('Static label previews rendered; no simulation or encoding.')
        return
    assert not OUT.exists(), 'Preserve the completed labeled export'
    partial = OUT.with_suffix('.partial.mp4')
    args = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
            '-pixel_format','rgb24','-video_size','1280x720','-framerate',str(FPS),'-i','pipe:0',
            '-an','-c:v','libx264','-threads','2','-preset','medium','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
    timeline, total = [], 0
    with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
        def emit(im):
            nonlocal total
            encoder.stdin.write(im.tobytes())
            total += 1
        for _ in range(6*FPS):
            emit(card())
        for i in STAGES:
            row=rows[i];native=indexed[row['name']];path=DATA/native['relative_path']
            assert path.stat().st_size==native['bytes'] and sha(path)==native['sha256']
            source_frames=[];cuts=[];start_frame=total;base=template(i,row)
            with h5py.File(path,'r') as f:
                assert f.attrs['status']=='complete' and f.attrs['image_encoding']=='jpeg95'
                demo=f['data/demo_0'];times=demo['next_time'][:]
                previous_end=None
                for lo,hi,rate in segments(i,row,times):
                    assert hi>lo and rate>0
                    begin=total;is_cut=previous_end is not None and abs(lo-previous_end)>.075
                    for n,t in enumerate(np.arange(lo,hi,rate/FPS)):
                        k=int(np.argmin(np.abs(times-t)))
                        emit(compose(base,demo,k,float(times[k]),rate,cut=is_cut and n<12))
                        source_frames.append(k)
                    cuts.append(dict(source_start_s=lo,source_end_s=hi,playback_rate=rate,
                                     output_start_s=begin/FPS,output_end_s=total/FPS,cut_before=is_cut))
                    previous_end=hi
                k=len(times)-1
                final=compose(base,demo,k,float(times[-1]),1.,frozen=True)
                for _ in range(2*FPS):
                    emit(final);source_frames.append(k)
            timeline.append(dict(name=row['name'],display_test=i+1,strict_success=row['surface_goal_zero_spill_success'],
                title=labels(i,row)[0],reason=labels(i,row)[1],detail=labels(i,row)[2],
                source_relative_path=native['relative_path'],source_sha256=native['sha256'],
                output_start_s=start_frame/FPS,output_end_s=total/FPS,segments=cuts,
                source_frame_indices=source_frames))
            print(f'Encoded test {i+1:02}: {total/FPS:.2f}s total',flush=True)
        outro_start=total/FPS
        for _ in range(7*FPS):
            emit(card(True))
        encoder.stdin.close()
        assert encoder.wait()==0
    stream=read_probe(partial)
    assert stream['width']==1280 and stream['height']==720 and int(stream['nb_read_frames'])==total
    assert stream['r_frame_rate']=='20/1' and abs(float(stream['duration'])-total/FPS)<.002
    subprocess.run(['ffmpeg','-v','error','-i',str(partial),'-f','null','-'],check=True)
    partial.replace(OUT)
    result=dict(video=OUT.name,video_sha256=sha(OUT),video_stream=stream,duration_s=total/FPS,
        final_report_sha256=sha(E/'final-benchmark.json'),checkpoint_sha256=report['checkpoint_sha256'],
        selection='Two prespecified successful examples (00/01), then all seven strict failures. Selected after evaluation for explanation; not a frequency-representative sample.',
        full_batch_successes=23,full_batch_tests=30,full_batch_spill_runs=2,display_test_numbers='One-based 01–30; native names are zero-based.',
        composition='Original paired task/wrist JPEGs, resized without changing their content. No generated camera frames. Playback rates and cuts are labeled; final frames held for 2 s. Silent H.264 video.',
        rendering='Local Mac CPU only; two ffmpeg threads; no Ubuntu connection, simulator restart, inference or training.',
        intro_end_s=6.,outro_start_s=outro_start,cases=timeline,
        dependencies=dict(python=sys.version.split()[0],h5py=h5py.__version__,numpy=np.__version__),
        full_decode_passed=True)
    (E/'labeled-highlights-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('video','video_sha256','duration_s','full_decode_passed')},indent=2))


def read_probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration',
        '-of','json',str(path)]))['streams'][0]


if __name__=='__main__':
    main(preview='--preview' in sys.argv)
