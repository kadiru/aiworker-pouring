import argparse
from pathlib import Path
import sys
import unittest
import torch
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--study", choices=("wide-mixed-r27","wide-mixed-r27b", "wide-mixed-r28"), default="wide-mixed-r27")
sys.path.insert(0,str(root/parser.parse_args().study/"src"))
torch.set_num_threads(2)
names = ("test_pour_size_autonomy","test_pour_temporal_localization",
         "test_pour_settled","test_pour_size","test_pour_guard")
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
