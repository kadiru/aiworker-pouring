from pathlib import Path
import sys
import unittest
import torch
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/"wide-mixed-r27"/"src"))
torch.set_num_threads(2)
names = ("test_pour_size_autonomy","test_pour_temporal_localization",
         "test_pour_settled","test_pour_size","test_pour_guard")
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
