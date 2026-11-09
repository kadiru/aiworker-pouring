"""Load USD schemas without SimulationApp and verify scoped support material."""
import hashlib,importlib.util,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
usd=next(Path('/isaac-sim/extscache').glob('omni.usd.libs-*'))
physx=next(Path('/isaac-sim/extscache').glob('omni.usd.schema.physx-*'))
if not os.environ.get('AIWORKER_SUPPORT_USD_BOOTSTRAP'):
    env=dict(os.environ,AIWORKER_SUPPORT_USD_BOOTSTRAP='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env['LD_LIBRARY_PATH']=str(usd/'bin')+':'+str(physx/'bin')+':'+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
sys.path[:0]=[str(ROOT/'staged'),str(usd),str(physx),'/workspace/src']
from pxr import Plug
for path in physx.glob('plugins/*/resources/plugInfo.json'):Plug.Registry().RegisterPlugins(str(path))
test=ROOT/'test_rigid_support_native.py';spec=importlib.util.spec_from_file_location('support_checks',test)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
report=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
    passed=result.wasSuccessful() and result.testsRun==3 and not result.skipped,
    test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((ROOT/'staged/pour_rigid_support.py').read_bytes()).hexdigest(),
    scope='CPU-only USD binding scope, legacy no-op and invalid-parameter checks. No SimulationApp or native dynamics.')
destination=ROOT/'support-cpu-checks.json';assert not destination.exists();destination.write_text(json.dumps(report,indent=2)+'\n')
assert report['passed'];print(json.dumps(report))
