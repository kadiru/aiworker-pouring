"""Freeze a new development tracker before evaluation; preserve all old results."""
import argparse,json,urllib.request
from prepare_dynamic_rim import ROOT,read,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--round',choices=('t0','t1'),default='t0');a=p.parse_args()
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'component-selection.json').exists()
    m=read(ROOT/'candidate.json');assert not m['autonomy_accepted']
    from fit_dynamic_rim import protected_unchanged
    protected_unchanged()
    assert read(ROOT/'coarse-d2.json')['passed_for_refinement'] and not read(ROOT/'rim-d2.json')['passed']
    files=['protocol.json','plan.json','candidate.json','cup-neural-localization.pt','coarse-d2.pt','coarse-d2.json','rim-d2.pt','rim-d2.json','rim-temporal-feasibility.json','dynamic_cup_tracker.py','dynamic_tracker_data.py','evaluate_dynamic_tracker.py','fit_dynamic_rim.py','scale_rim_refiner.py','scale_data_admission.py',m['collection_audit'],m['collection_review']]
    for part in ('seed_pilot','seed_remaining','low_train','train','validation'):files += [part+'-r0-audit.json',part+'-r0-manual-review.json']
    settings=dict(shifts=[[x,y] for x in (-2,0,2) for y in (-2,0,2)],max_refinement_pixels=32.,association_pixels=24.,search_pixels=16,minimum_similarity=.75,maximum_movement_mm=1.5)
    if a.round=='t1':
        assert not read(ROOT/'tracker-t0.json')['passed']
        files+=['tracker-t0.json','tracker-t0-protocol.json','tracker-t0-rejection-preview.json']
        settings.update(reference_vertical_fraction=[.05,.85],template_parts=3,minimum_support=2)
    config=dict(round=a.round,settings=settings,frozen_files={n:sha(ROOT/n) for n in sorted(set(files))},gates=read(ROOT/'protocol.json')['gates'],
        scope='Stationary known-size cup only. Equal old/d2 fine predictions, nine crop translations averaged per model, eight-frame causal confirmation, fixed visual lower-outline reference. No pose/level inference input. Reject absence, association loss, visual mismatch or >1.5 mm template displacement. No optimizer updates or reserved data.',
        accuracy_policy='Retain 3 mm on every initial raw prediction (all 12 frames) and every confirmed tracked output; confirmation required within 12 frames in every episode. All former sampled frames retained in chronological order. Initial unconfirmed frames cannot authorize control. Synthetic erased-cup and +/-3.5 mm view-translation challenges must reject; CPU/GPU status and <0.1 pixel coordinate parity required. No active controller acceptance.')
    with (ROOT/('tracker-'+a.round+'-protocol.json')).open('x') as f:json.dump(config,f,indent=2);f.write('\n')
    print(json.dumps(dict(frozen=True,round=a.round,autonomy_accepted=False)))
if __name__=='__main__':main()
