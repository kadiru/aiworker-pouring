"""Historical one-off v4 baseline-scope selection; superseded by scoped refits."""
import hashlib,json,shutil,sys,time
from pathlib import Path
import torch
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
from fit_perception_locator import collect,validate,read,sha,ROOT
from pour_cup_locator import CupLocator
def main():
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'scope-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*PV_A_perception_test_*/parameters.json'))
    assert not list(Path('/workspace/outputs/fountain').glob('*/*PV_A_development_*/parameters.json'))
    report=read(ROOT/'cup-neural-localization.json');assert not report['passed']
    assert sha(ROOT/'cup-neural-localization-attempt.pt')==report['attempt_checkpoint_sha256']
    candidate=torch.load(ROOT/'cup-neural-localization-attempt.pt',map_location='cpu',weights_only=False)
    plan,data,cal=collect();assert plan['version']==4,'Historical selector must not overwrite a newer plan';torch.set_num_threads(4)
    names={s['name'] for s in plan['collect'] if s['cup_height_scale']==s['cup_diameter_scale']==1.}
    model=CupLocator().cuda().eval();model.load_state_dict(candidate['state']);train=validate(model,data['train'])
    validation=candidate['validation'];positive=[r for r in validation['rows'] if r['name'] in names]
    training=[r for r in train['rows'] if r['name'] in names]
    assert len(positive)==12 and len(training)==9
    assert max(r['error_mm'] for r in positive+training)<plan['component_gates']['localization_max_error_mm']
    minimum=min(r['score'] for r in positive+training)
    # Keep absence evidence from every native size, even though positive scope
    # is restricted to the baseline vessel.
    negatives=[r['synthetic_absent_score'] for r in validation['rows']+train['rows'] if r['domain']=='native_detail']
    maximum=max(negatives);assert minimum-maximum>plan['component_gates']['localization_negative_margin']
    threshold=(minimum+maximum)/2
    identity=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(candidate['state'].items()))).hexdigest()
    selected=dict(bank=dict(kind='cup_heatmap',state=candidate['state'],identity=identity),calibration=cal,threshold=threshold,
        consensus=dict(window=20,votes=8,radius_px=1.5,fraction=.6,score_floor=max(.01,threshold*.6)))
    torch.save(selected,ROOT/'cup-neural-localization.pt')
    selection=dict(passed=True,scope='baseline_cup_only',height_scale=1.,diameter_scale=1.,epoch=candidate['epoch'],
        max_validation_error_mm=max(r['error_mm'] for r in positive),max_training_error_mm=max(r['error_mm'] for r in training),
        minimum_positive_score=minimum,maximum_synthetic_absent_score=maximum,threshold=threshold,
        checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt'),rejected_all_size_report_sha256=sha(ROOT/'cup-neural-localization.json'),
        validation=positive,training=training,source_sha256=sha(Path(__file__)),
        rationale='Repair original-vessel autonomy first. All-size validation remains failed and resized autonomy stays gated. Same 3 mm and absence-separation thresholds, narrower declared scope selected before reserved evaluation; no reserved images used.')
    archive=ROOT/'before-baseline-scope';archive.mkdir()
    for n in ('plan.json','candidate.json'):shutil.copy2(ROOT/n,archive/n)
    plan.update(version=5,primary_perception_scope='baseline_cup_only',resized_cup_autonomy='gated; all-size localization validation failed')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest=read(ROOT/'candidate.json');manifest['plan_sha256']=sha(ROOT/'plan.json');manifest['validated_cup_scope']='baseline_only'
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'scope-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({k:v for k,v in selection.items() if k not in ('training','validation','rationale')}))
if __name__=='__main__':main()
