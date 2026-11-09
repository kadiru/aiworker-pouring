"""Finite read-only live overview capture; formal review uses audited HDF5."""
import argparse,json,time,urllib.request
from pathlib import Path
BASE='http://192.168.68.60:8212'
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(exist_ok=False);records=[];deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    h=json.load(urllib.request.urlopen(BASE+'/health',timeout=10))
    assert not h.get('error')
    name=(h.get('trial') or {}).get('name');phase=h.get('phase','')
    if name!=a.case:
        if records:break
        raise RuntimeError('The requested attempt is no longer active; no replay is requested')
    if 'Returning upright' in phase:
        m=h['metrics'];t=m['t']
        if not records or t-records[-1]['t']>=.35:
            dest=a.output/f'return-{len(records):02}.jpg'
            dest.write_bytes(urllib.request.urlopen(BASE+'/camera/overview.jpg',timeout=10).read())
            r=dict(t=t,phase=phase,pose=m['bottle_pose_xyz'],tilt=m['bottle_tilt_deg'],path=str(dest))
            records.append(r);print(json.dumps(r),flush=True)
    elif records:break
    time.sleep(2)
else:raise TimeoutError('Finite read-only capture expired')
(a.output/'capture.json').write_text(json.dumps(dict(case=a.case,frames=records,
    scope='Live overview snapshots; health/image requests may straddle one camera frame. Not quantitative validation or a replacement for audited recording review.'),indent=2)+'\n')
print('LIVE_RETURN_CAPTURE_COMPLETE',flush=True)
