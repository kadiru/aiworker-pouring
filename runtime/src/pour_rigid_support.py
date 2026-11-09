"""Opt-in dry rigid support friction on the bottle base only.

The wall/gripper and particle materials are untouched. PhysX 107.3 particles
use their own PBD dynamic friction, without rigid material combine modes.
"""
import math

def apply(stage,spec):
    config=spec.get('bottle_support_friction')
    if config is None:return None
    static=float(config['static']);dynamic=float(config['dynamic'])
    if not (math.isfinite(static) and math.isfinite(dynamic) and 0<=dynamic<=static<=1):
        raise ValueError('Bottle support friction requires 0 <= dynamic <= static <= 1')
    from pxr import UsdPhysics,UsdShade,PhysxSchema
    base=stage.GetPrimAtPath('/World/Bottle/Base')
    assert base and base.HasAPI(UsdPhysics.CollisionAPI)
    material=UsdShade.Material.Define(stage,'/World/BottleSupportContact')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static);api.CreateDynamicFrictionAttr(dynamic);api.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('max')
    UsdShade.MaterialBindingAPI.Apply(base).Bind(material,materialPurpose='physics')
    return dict(static=static,dynamic=dynamic,combine='max',collider='/World/Bottle/Base',
        scope='Authored rigid support contact; not a measured glass/table coefficient. No wall, finger, particle or render-material changes.')
