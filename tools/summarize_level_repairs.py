"""Compact Git evidence; full per-frame/epoch reports stay preserved natively."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/dynamic-A'
def main():
    p=argparse.ArgumentParser();p.add_argument('rounds',nargs='+');a=p.parse_args()
    for round in a.rounds:
        assert round.startswith('r') and round[1:].isdigit()
        for kind in ('rgb','color'):
            path=ROOT/f'pair-{round}-{kind}.json';raw=json.loads(path.read_text());episodes={}
            for r in raw['rows']:
                key=(r['split'],r['name']);episodes.setdefault(key,[]).append(r)
            rows=[]
            for (split,name),values in sorted(episodes.items()):
                errors=[v['error'] for v in values]
                rows.append(dict(split=split,name=name,frames=len(values),target_range=[min(v['target'] for v in values),max(v['target'] for v in values)],mae=sum(abs(e) for e in errors)/len(errors),mean_error=sum(errors)/len(errors),max_abs=max(abs(e) for e in errors),source_sha256=values[0]['source_sha256']))
            result={k:v for k,v in raw.items() if k not in ('rows','history')}
            result.update(episodes=rows,history=[{k:v for k,v in h.items() if k not in ('training','validation','per_episode_mae')} for h in raw['history']],
                full_native_report='/workspace/learning/A-scale-dynamic-20261109-v1/'+path.name,full_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Compact evidence from the complete unchanged native report. All selected metrics retained; per-frame rows and detailed epoch metrics remain in the SHA-bound original. No refitting or rescoring.')
            (ROOT/f'pair-{round}-{kind}-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Compact reports written for '+', '.join(a.rounds))
if __name__=='__main__':main()
