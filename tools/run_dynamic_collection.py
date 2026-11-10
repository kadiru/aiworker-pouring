"""Sequential finite calibration/audit pipeline; never fits or runs autonomy."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-dynamic-20261109-v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('parts',nargs='+',choices=('seed_remaining','low_train','train','validation'));a=p.parse_args()
    for part in a.parts:
        w.isolation()
        subprocess.run([sys.executable,str(ROOT/'watch_dynamic_stage.py'),'--part',part],check=True)
        subprocess.run(['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-scale-dynamic-20261109-v1/check_dynamic_stage.py','--part',part],check=True)
    print('CALIBRATION_AND_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
