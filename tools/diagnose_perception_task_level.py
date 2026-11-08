"""CPU replay of closed autonomous stop views; never alters control or scoring."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path[:0] = ['/workspace/src', '/workspace']
from validate_pour_recording import h5py, decode
from pour_guard_models import CupEstimator
from pour_guard_vision import cup_crop, color_geometry_readout
ROOT = Path('/workspace/learning/A-perception-20261108-v1')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--name', required=True); a = p.parse_args()
    assert a.name.startswith('PV_A_development_') and '/' not in a.name
    paths = list(Path('/workspace/outputs/fountain').glob('*/*'+a.name+'-*/episode.hdf5'))
    assert len(paths) == 1
    path = paths[0]; spec = read(path.parent/'parameters.json')
    samples = read(path.parent/'samples.json'); decisions = read(path.parent/'policy-decisions.json')
    assert read(path.parent/'metrics.json')['policy']['checkpoint_sha256'] == sha(ROOT/'best.pt')
    out = ROOT/(a.name+'-level-diagnostic.json'); assert not out.exists()
    stop = next(i for i,d in enumerate(decisions) if d['stop_latched'])
    assert decisions[stop]['cup_xy'] is not None and stop >= 6
    torch.set_num_threads(2)
    saved = torch.load(ROOT/'best.pt', map_location='cpu', weights_only=False)
    model = CupEstimator().eval(); model.load_state_dict(saved['cup_model'])
    cal = copy.deepcopy(saved['cup_detail_calibration']); rows = []
    selected = list(range(stop-5, stop+1)) + [len(decisions)-1]
    sheet = Image.new('RGB', (1024, 480*len(selected)), (245,245,245)); draw = ImageDraw.Draw(sheet)
    with h5py.File(path) as f:
        assert f.attrs['status'] == 'complete'; ds = f['data/demo_0']
        for row_number, i in enumerate(selected):
            d = decisions[i]; observed = samples[i-1]
            assert abs(float(ds['time'][i])-observed['t']) < .0002
            rgb = decode(ds['obs/cup_detail_image'][i], 'jpeg95')
            xy = np.array(d['cup_xy']); variants = []
            for label, center in [('runtime',xy), ('truth_diagnostic',np.array(spec['cup_world'][:2]))] + [
                (f'offset_{dx}_{dy}',xy+[dx,dy]) for dx,dy in [(-.001,0),(.001,0),(0,-.001),(0,.001)]]:
                crop = torch.from_numpy(cup_crop(rgb,center,cal))[None]
                cnn = float(np.clip(float(model(crop)[0])+saved['cup_bias'],0,1))
                color = float(color_geometry_readout(crop,saved['surface_geometry'],center)[0].clamp(0,1))
                variants.append(dict(label=label,cnn=cnn,color=color,error=cnn-observed['cup_fill_fraction']))
            assert abs(variants[0]['cnn']-d['cup_predicted_fraction']) < 1e-4
            assert abs(variants[0]['color']-d['surface_geometry_fraction']) < 1e-4
            rows.append(dict(index=i,time_s=d['t'],mode=d['mode'],pause_count=d['pause_count'],
                observed_height=observed['cup_fill_fraction'],column_height=observed.get('cup_column_height_fraction'),
                cup_nominal_ml=observed['cup_ml'],source_nominal_ml=observed['bottle_ml'],
                localization_error_mm=float(np.linalg.norm(xy-spec['cup_world'][:2])*1000),variants=variants))
            y=row_number*480
            draw.text((8,y+8),f'{a.name} | {i} | {d["t"]:.3f}s | {d["mode"]}',fill='black')
            draw.text((8,y+28),f'Raw observed {observed["cup_fill_fraction"]:.4f} | CNN {variants[0]["cnn"]:.4f} | Color {variants[0]["color"]:.4f}',fill='black')
            sheet.paste(Image.fromarray(rgb).resize((420,420)),(0,y+52))
            crop=cup_crop(rgb,xy,cal).astype(np.uint8)
            sheet.paste(Image.fromarray(crop).resize((384,384)),(430,y+52))
    image_path=ROOT/(a.name+'-level-diagnostic.jpg');sheet.save(image_path,quality=94)
    last=samples[-20:]
    report=dict(name=a.name,source_sha256=sha(path),decisions_sha256=sha(path.parent/'policy-decisions.json'),
        samples_sha256=sha(path.parent/'samples.json'),checkpoint_sha256=sha(ROOT/'best.pt'),
        helper_sha256=sha(Path(__file__)),rows=rows,image=dict(path=str(image_path),sha256=sha(image_path)),
        final_second=[{k:s.get(k) for k in ('t','cup_fill_fraction','cup_column_height_fraction','cup_ml','bottle_ml','spilled_ml','ever_table_ml','phase')} for s in last],
        scope='Closed native observations replayed on CPU. Raw prior-step surface labels align with each input image. True cup center and particle-derived heights are offline diagnostics only. No fitting, live truth in control, or replacement of final-sample scores.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(name=a.name,rows=rows,final_height=samples[-1]['cup_fill_fraction'])))
if __name__ == '__main__': main()
