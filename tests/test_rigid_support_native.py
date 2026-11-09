"""USD material-scoping checks; native physics is validated separately."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
from pour_rigid_support import apply

class SupportMaterial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,PhysxSchema
        except ImportError as error:raise unittest.SkipTest('Native USD dependency: '+str(error))
        cls.Usd,cls.UsdGeom,cls.UsdShade,cls.UsdPhysics,cls.PhysxSchema=Usd,UsdGeom,UsdShade,UsdPhysics,PhysxSchema

    def fixture(self):
        stage=self.Usd.Stage.CreateInMemory();wet=self.UsdShade.Material.Define(stage,'/World/WetContact')
        api=self.UsdPhysics.MaterialAPI.Apply(wet.GetPrim());api.CreateStaticFrictionAttr(0.);api.CreateDynamicFrictionAttr(0.)
        for path in ('/World/Bottle/Base','/World/Bottle/Collision/Wall0_0','/World/Cup/Base','/World/Bench'):
            prim=self.UsdGeom.Cube.Define(stage,path).GetPrim();self.UsdPhysics.CollisionAPI.Apply(prim)
            self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(wet,materialPurpose='physics')
        return stage

    def binding(self,stage,path):
        return str(self.UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(materialPurpose='physics')[0].GetPath())

    def test_only_base_binding_changes(self):
        stage=self.fixture();result=apply(stage,dict(bottle_support_friction=dict(static=.4,dynamic=.3)))
        self.assertEqual(self.binding(stage,'/World/Bottle/Base'),'/World/BottleSupportContact')
        for path in ('/World/Bottle/Collision/Wall0_0','/World/Cup/Base','/World/Bench'):
            self.assertEqual(self.binding(stage,path),'/World/WetContact')
        prim=stage.GetPrimAtPath('/World/BottleSupportContact');api=self.UsdPhysics.MaterialAPI(prim)
        self.assertAlmostEqual(api.GetStaticFrictionAttr().Get(),.4);self.assertAlmostEqual(api.GetDynamicFrictionAttr().Get(),.3)
        self.assertEqual(self.PhysxSchema.PhysxMaterialAPI(prim).GetFrictionCombineModeAttr().Get(),'max')
        self.assertEqual(result['collider'],'/World/Bottle/Base')

    def test_legacy_scene_has_no_new_material(self):
        stage=self.fixture();self.assertIsNone(apply(stage,{}))
        self.assertFalse(stage.GetPrimAtPath('/World/BottleSupportContact'))
        self.assertEqual(self.binding(stage,'/World/Bottle/Base'),'/World/WetContact')

    def test_invalid_coefficients_fail_before_scene_mutation(self):
        for static,dynamic in ((.2,.4),(float('nan'),.2),(.4,-.1)):
            stage=self.fixture()
            with self.assertRaises(ValueError):apply(stage,dict(bottle_support_friction=dict(static=static,dynamic=dynamic)))
            self.assertFalse(stage.GetPrimAtPath('/World/BottleSupportContact'))
if __name__=='__main__':unittest.main()
