"""Focused CPU contracts in the native dependency environment."""
import json,unittest
from fit_dynamic_rim import ROOT,sha
import test_perception_repair_data,test_dynamic_cup_tracker,test_dynamic_holdout_metrics,test_level_repair_gates

def main():
    modules=[test_perception_repair_data,test_dynamic_cup_tracker,test_dynamic_holdout_metrics,test_level_repair_gates]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                sources={m.__name__+'.py':sha(ROOT/(m.__name__+'.py')) for m in modules},
                scope='CPU regression contracts, not simulation task success.')
    with (ROOT/'expanded-perception-cpu-checks.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    assert result.wasSuccessful()

if __name__=='__main__':main()
