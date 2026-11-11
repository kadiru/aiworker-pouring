"""Draft and geometrically screen added training data; no simulation launch."""
import subprocess,sys
from fit_dynamic_rim import ROOT
def main():
    subprocess.run([sys.executable,str(ROOT/'prepare_midfill_collection.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'preflight_dynamic_geometry.py'),'--part','train','--protocol','protocol-midfill-v6-draft.json'],check=True)
if __name__=='__main__':main()
