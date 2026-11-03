"""Camera-visible vessel appearance, authored independently of the physics shapes."""
import math
from pxr import Gf, Sdf, UsdGeom, UsdShade

PRESETS = {
    'A': {'opacity': .08, 'roughness': .20},
    'B': {'opacity': .14, 'roughness': .35},
    'C': {'opacity': .20, 'roughness': .50},
}


def band(stage, vessel, profile, low, high, material):
    points = []
    for z in (low, high):
        radius = profile[0][1] + (profile[1][1]-profile[0][1]) * (
            z-profile[0][0]) / (profile[1][0]-profile[0][0]) + .00252
        for i in range(20):
            angle = 2*math.pi*i/20
            points.append((radius*math.cos(angle), radius*math.sin(angle), z))
    mesh = UsdGeom.Mesh.Define(stage, vessel+'/VisualBand')
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4]*20)
    mesh.CreateFaceVertexIndicesAttr([
        vertex for i in range(20) for vertex in
        (i, (i+1)%20, 20+(i+1)%20, 20+i)])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def apply(stage, frosted=True, variant='A'):
    preset = PRESETS[variant]
    from bottle_pour import BOTTLE, CUP
    marker = UsdShade.Material.Define(stage, '/World/VesselBandMaterial')
    shader = UsdShade.Shader.Define(stage, str(marker.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.025,.14,.19))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.75)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    marker.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for path in ('/World/ClearBottle', '/World/ClearTumbler'):
        material = UsdShade.Material.Get(stage, path)
        if not material:
            continue
        material.GetPrim().RemoveProperty('outputs:mdl:surface')
        if frosted:
            surface = UsdShade.Shader.Define(stage, path+'/LightFrost'+variant)
            surface.CreateIdAttr('UsdPreviewSurface')
            surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.88,.91,.92))
            surface.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(preset['opacity'])
            surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(preset['roughness'])
            surface.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.5)
            surface.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        else:
            surface = UsdShade.Shader.Get(stage, path+'/Shader')
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    for vessel, profile, low, high in (
        ('/World/Bottle', BOTTLE, .068, .074),
        ('/World/Cup', CUP, .029, .031)):
        if not stage.GetPrimAtPath(vessel):
            continue
        band(stage, vessel, profile, low, high, marker)
        UsdGeom.Imageable(stage.GetPrimAtPath(vessel+'/VisualBand')).CreateVisibilityAttr().Set(
            'inherited' if frosted else 'invisible')
