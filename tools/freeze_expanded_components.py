"""Once-only reserved permission only after every expanded development gate."""
import argparse,json
from pathlib import Path
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('r8','r9','r10','r11','r12'),default='r8');args=parser.parse_args();prefix='pair-'+args.round
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    protected_unchanged();protocol=read(ROOT/'protocol.json');cfg=read(ROOT/(prefix+'-protocol.json'))
    assert read(ROOT/'expanded-perception-development.json')['tracker_passed']
    assert read(ROOT/'expanded-perception-cpu-checks.json')['passed']
    assert read(ROOT/(prefix+'.json'))['passed']
    assert cfg['protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in cfg['frozen_files'].items())
    tracker_prefix='tracker-t3' if args.round=='r12' else 'tracker-t2'
    tracker=read(ROOT/(tracker_prefix+'.json'))
    count=279 if args.round=='r12' else 243
    assert tracker['passed'] and len(tracker['cases'])==count and all(r['passed'] for r in tracker['cases'])
    assert len(tracker['synthetic_challenges'])==180 and all(r['rejected'] for r in tracker['synthetic_challenges'])
    assert tracker['checkpoint_sha256']==sha(ROOT/(tracker_prefix+'.pt'))
    cache=read(ROOT/('level-cache-v6.json' if args.round=='r12' else 'level-cache-v4.json'));assert cache['passed'] and len(cache['admissions'])==count
    development=None
    if args.round in ('r10','r11'):
        extra=read(ROOT/'midfill-tracker-admission.json')
        assert extra['passed'] and len(extra['cases'])==36 and all(r['passed'] for r in extra['cases'])
        development='midfill-development.json' if args.round=='r10' else 'physical-crop-development.json'
        assert read(ROOT/development)['passed']
        assert read(ROOT/'level-cache-v5.json')['passed']
        if args.round=='r11':assert read(ROOT/'physical-crop-contracts.json')['passed']
    if args.round=='r12':
        development='relocalized-midfill-development.json'
        assert read(ROOT/development)['passed'] and read(ROOT/development)['tracker_passed']
        assert read(ROOT/'train-r3-manual-review.json')['accepted']
    for branch in ('rgb','color'):
        r=read(ROOT/(prefix+'-'+branch+'.json'))
        assert r['passed'] and r['selected']['passed']
        assert r['selected']['max_episode_mae']<protocol['gates']['level_validation_mae']
        assert r['checkpoint_sha256']==sha(ROOT/(prefix+'-'+branch+'.pt'))
    files=set(cfg['frozen_files'])|{prefix+'.json',prefix+'-protocol.json',prefix+'-rgb.pt',prefix+'-color.pt',
        prefix+'-rgb.json',prefix+'-color.json','expanded-perception-development.json','expanded-perception-cpu-checks.json',
        'evaluate_dynamic_holdout.py','freeze_expanded_components.py'}
    if development is not None:files.add(development)
    selection=dict(all_required_development_gates_passed=True,protocol_sha256=sha(ROOT/'protocol.json'),
                   locator=tracker_prefix+'.pt',rgb=prefix+'-rgb.pt',color=prefix+'-color.pt',
                   holdout_names=[s['name'] for s in protocol['holdout']],
                   evaluation_helper_sha256=sha(ROOT/'evaluate_dynamic_holdout.py'),
                   frozen_files={n:sha(ROOT/n) for n in sorted(files)},autonomy_accepted=False,
                   scope='18 once-only known-size moving perception fixtures. No autonomous pouring acceptance; no refitting after test observation.')
    assert len(selection['holdout_names'])==18
    with (ROOT/'component-selection.json').open('x') as f:json.dump(selection,f,indent=2);f.write('\n')
    print(json.dumps(dict(frozen=True,reserved_scenes=18,autonomy_accepted=False)),flush=True)

if __name__=='__main__':main()
