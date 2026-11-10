"""Finite-source pouring fixture and a state-based reference controller for AIWorker."""
import math
from collections import deque
import numpy as np

# Height and inner circumradius in metres. The bottle has one open mouth.
BOTTLE = ((.004, .025), (.082, .025), (.107, .009), (.135, .009))
CUP = ((.004, .020), (.038, .024))
SIDES = 20
GRASP_Z = .048
START_X = -.115
MOUTH_X = -.013
MOUTH_Z = .054
LIP = BOTTLE[-1][0]
# Nominal particle volume retained at the rim in the first measured long pour.
BRIM_NOMINAL_ML = 72.8
RETURN_FLOW_SECONDS = .20
CAPACITY_ML = (SIDES * math.sin(2 * math.pi / SIDES) / 2
               * (CUP[-1][0] - CUP[0][0]) / 3
               * (CUP[0][1]**2 + CUP[0][1]*CUP[-1][1] + CUP[-1][1]**2) * 1e6)


def bottle_spec(base, small):
    return dict(base, name='bottle_short_fill' if small else 'bottle_full_cup',
                protocol=5, bottle=True, initial_ml=20. if small else 100.,
                slosh_speed=0., hinge=False, obstacles=False, max_seconds=15.,
                opacity=.78, roughness=.09, appearance='default',
                stop_teacher_version=2, brim_nominal_ml=BRIM_NOMINAL_ML,
                target_fraction=.87, return_flow_seconds=RETURN_FLOW_SECONDS)


def stop_signal(cup_ml, left_ml, height, inbound_ml, flow_ml_s, initial_ml,
                target_fraction=.87, return_flow_seconds=RETURN_FLOW_SECONDS, brim_ml=BRIM_NOMINAL_ML):
    runoff = min(left_ml, max(0., flow_ml_s)*return_flow_seconds)
    predicted = cup_ml + inbound_ml + runoff
    if cup_ml >= .55*brim_ml and height >= .55 and predicted >= target_fraction*brim_ml:
        return 'cup_near_full', predicted
    if left_ml <= max(.12, .005*initial_ml):
        return 'bottle_empty', predicted
    return None, predicted


def ease(x):
    u = min(max(float(x), 0.), 1.)
    return u*u*(3-2*u)


def pose(angle, lift, start_xy=None, mouth_z=MOUTH_Z):
    radians = math.radians(angle)
    sine, cosine = math.sin(radians), math.cos(radians)
    fraction = ease(angle / 70.)
    sx, sy = (START_X, 0.) if start_xy is None else start_xy
    mouth_x = sx + (MOUTH_X - sx)*fraction
    # Keep the bottom clear of the bench before placing the mouth just above the rim.
    mouth_z = max(mouth_z, LIP*cosine + (.027*abs(sine) + .026)*lift)
    base = np.array([mouth_x - LIP*sine, sy*(1-fraction), mouth_z - LIP*cosine])
    rotation = np.array([[cosine, 0., sine], [0., 1., 0.], [-sine, 0., cosine]])
    return base, rotation


def initial_points(spec):
    spacing = spec['spacing']
    radius = BOTTLE[0][1]*math.cos(math.pi/SIDES) - 2*spacing
    n = int(radius/spacing)
    cross = sorted(((i*spacing, j*spacing) for i in range(-n,n+1) for j in range(-n,n+1)
                    if (i*i+j*j)*spacing**2 < radius**2), key=lambda xy: xy[0]**2+xy[1]**2)
    count = round(spec['initial_ml']/(spacing**3*1e6))
    if count==0:
        if spec.get('calibration_only') and spec.get('calibration_motion') and not spec.get('policy_evaluation'):
            return np.empty((0,3),dtype=np.float32)
        raise ValueError('An empty initial bottle is supported only by authored motion calibration')
    pts = np.array([[*cross[i%len(cross)], BOTTLE[0][0]+2*spacing+(i//len(cross))*spacing]
                    for i in range(count)], dtype=np.float32)
    if pts[:,2].max() > BOTTLE[1][0] - spacing:
        raise ValueError('Initial fill exceeds bottle body')
    pts[:,0] += START_X
    return pts


def vessel_mask(points, profile):
    z = points[:,2]
    radius = np.interp(z, [v[0] for v in profile], [v[1] for v in profile])
    azimuth = np.arctan2(points[:,1], points[:,0])
    # Radial boundary of the actual polygon between consecutive vertex angles.
    side_angle = (azimuth % (2*math.pi/SIDES)) - math.pi/SIDES
    bound = radius * math.cos(math.pi/SIDES) / np.cos(side_angle)
    return ((z >= profile[0][0]-.0003) & (z < profile[-1][0]) &
            (np.hypot(points[:,0],points[:,1]) <= bound+.0003))


def _mesh(stage, path, points, faces):
    from pxr import UsdGeom
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def _collider(prim, convex=False):
    from pxr import UsdPhysics, PhysxSchema, Sdf
    from omni.physx.scripts import physicsUtils
    UsdPhysics.CollisionAPI.Apply(prim)
    if convex:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(.001)
    api.CreateRestOffsetAttr(0.)
    physicsUtils.add_physics_material_to_prim(prim.GetStage(), prim, Sdf.Path('/World/WetContact'))


def _vessel(stage, path, profile, material, moving):
    from pxr import UsdGeom, UsdShade, UsdPhysics, Gf
    root = UsdGeom.Xform.Define(stage, path)
    position = root.AddTranslateOp()
    position.Set(Gf.Vec3d(START_X if moving else 0., 0., 0.))
    orientation = root.AddOrientOp()
    orientation.Set(Gf.Quatf(1., Gf.Vec3f(0.)))
    if moving:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.18)
        UsdGeom.Xform.Define(stage, path+'/GraspFrame').AddTranslateOp().Set(Gf.Vec3d(0,0,GRASP_Z))
        UsdGeom.Xform.Define(stage, path+'/MouthFrame').AddTranslateOp().Set(Gf.Vec3d(0,0,LIP))
    thickness = .0024
    bottom = UsdGeom.Cylinder.Define(stage, path+'/Base')
    bottom.CreateRadiusAttr(profile[0][1]+thickness)
    bottom.CreateHeightAttr(profile[0][0])
    bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,profile[0][0]/2))
    UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(material)
    _collider(bottom.GetPrim())

    def vertex(radius, phi, z):
        return (radius*math.cos(phi), radius*math.sin(phi), z)

    # One continuous visual shell, with a separate convex collision piece per wall sector.
    shell_points, shell_faces = [], []
    contour = list(profile) + [(z, r+thickness) for z,r in reversed(profile)]
    for z, r in contour:
        shell_points.extend(vertex(r, 2*math.pi*i/SIDES, z) for i in range(SIDES))
    for ring in range(len(contour)):
        following = (ring+1) % len(contour)
        for i in range(SIDES):
            j = (i+1) % SIDES
            shell_faces.append((ring*SIDES+i, following*SIDES+i, following*SIDES+j, ring*SIDES+j))
    shell = _mesh(stage, path+'/Glass', shell_points, shell_faces)
    UsdShade.MaterialBindingAPI.Apply(shell.GetPrim()).Bind(material)
    faces = [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    for k, ((z0,r0),(z1,r1)) in enumerate(zip(profile,profile[1:])):
        for i in range(SIDES):
            lo = 2*math.pi*i/SIDES - .005
            hi = 2*math.pi*(i+1)/SIDES + .005
            points = [vertex(r+d, phi, z) for z,r in [(z0,r0),(z1,r1)]
                      for d in [0.,thickness] for phi in [lo,hi]]
            part = _mesh(stage, f'{path}/Collision/Wall{k}_{i}', points, faces)
            _collider(part.GetPrim(), convex=True)
            part.CreateVisibilityAttr('invisible')
    return position, orientation


class Pour:
    def __init__(self, stage, spec, bottle_glass, cup_glass):
        from pour_vessel_geometry import cup_profile, cup_geometry
        self.spec = spec
        self.cup_profile = cup_profile(spec)
        self.cup_geometry = cup_geometry(spec)
        self.position, self.orientation = _vessel(stage,'/World/Bottle',BOTTLE,bottle_glass,True)
        _vessel(stage,'/World/Cup',self.cup_profile,cup_glass,False)
        self.angle = 0.
        self.base, self.rotation = pose(0.,0.)
        self.stop_time = None
        self.stop_angle = 0.
        self.reason = None
        self.confirmed = 0
        self.last_signal = None
        self.source_history = deque(maxlen=4)
        self.phase = 'Settling initial fill'
        self.points = initial_points(spec)

    def drive(self, t):
        from pxr import Gf
        lift = ease((t-.65)/.5)
        if self.stop_time is None:
            self.angle = min(165., max(0., (t-1.15)*33.))
            self.phase = ('Settling initial fill' if t<=.65 else 'Lifting bottle' if t<=1.15
                          else 'Tilting to pour')
        else:
            elapsed = t-self.stop_time
            self.angle = self.stop_angle*(1-ease(elapsed/.85))
            lift = 1-ease((elapsed-.90)/.55)
            if elapsed<.90:
                self.phase = 'Returning upright'
            elif elapsed<1.45:
                self.phase = 'Setting bottle down'
            else:
                self.phase = 'Settling drink'
        self.base, self.rotation = pose(self.angle,lift)
        half = math.radians(self.angle)/2
        self.orientation.Set(Gf.Quatf(math.cos(half),Gf.Vec3f(0.,math.sin(half),0.)))
        self.position.Set(Gf.Vec3d(*self.base))

    def observe(self, p, v, t):
        if not (len(p)==len(v)==len(self.points) and np.isfinite(p).all() and np.isfinite(v).all()):
            raise RuntimeError('Finite particle inventory lost during bottle pour')
        local = (p-self.base) @ self.rotation
        source = vessel_mask(local,BOTTLE)
        profile = self.cup_profile
        cup = vessel_mask(p,profile) & ~source
        outside = ~(source | cup)
        airborne = outside & (p[:,2]>.0035) & (np.linalg.norm(v,axis=1)>.04)
        spilled = outside & ~airborne
        xyz = p[cup]
        height = 0.
        if len(xyz)>=40:
            sector = (np.arctan2(xyz[:,1],xyz[:,0])+math.pi)*8/(2*math.pi)
            tops = [np.quantile(xyz[(sector>=i)&(sector<i+1),2],.97) for i in range(8)
                    if np.count_nonzero((sector>=i)&(sector<i+1))>=5]
            if len(tops)>=6:
                height = float(np.clip((np.median(tops)+self.spec['spacing']/2-profile[0][0]) /
                                       (profile[-1][0]-profile[0][0]),0.,1.))
        ml = self.spec['spacing']**3*1e6
        left = int(source.sum())
        cup_ml = float(cup.sum()*ml)
        source_ml = left*ml
        self.source_history.append((t,source_ml))
        old_t, old_ml = self.source_history[0]
        flow_ml_s = max(0.,(old_ml-source_ml)/max(t-old_t,1e-6))
        inbound = airborne & (p[:,2]>=profile[-1][0]) & (v[:,2]<0.) & (np.hypot(p[:,0],p[:,1])<1.25*profile[-1][1])
        reason,predicted = stop_signal(cup_ml,source_ml,height,inbound.sum()*ml,flow_ml_s,len(p)*ml,
                                     self.spec.get('target_fraction',.87),
                                     self.spec.get('return_flow_seconds',RETURN_FLOW_SECONDS),
                                     self.cup_geometry['legacy_brim_nominal_ml'])
        if t<=1.15:
            reason=None
        self.confirmed = self.confirmed+1 if reason and reason==self.last_signal else int(reason is not None)
        self.last_signal = reason
        expert_enabled=(not self.spec.get('policy_evaluation') or getattr(self,'expert_active',False)) and not self.spec.get('calibration_only')
        return_allowance=3.4 if self.spec.get('expert_recovery') else 2.4
        if expert_enabled and self.stop_time is None and (self.confirmed>=2 or t>=self.spec['max_seconds']-return_allowance):
            self.reason = reason if self.confirmed>=2 else 'timeout'
            self.stop_time = t
            self.stop_angle = self.angle
        return dict(t=round(t,4),finite=True,count=len(p),bottle_ml=left*ml,
                    cup_ml=cup_ml,airborne_ml=float(airborne.sum()*ml),
                    spilled_ml=float(spilled.sum()*ml),cup_fill_fraction=height,
                    bottle_pose_xyz=self.base.tolist(),bottle_quat_xyzw=[0.,math.sin(math.radians(self.angle)/2),0.,math.cos(math.radians(self.angle)/2)],
                    bottle_tilt_deg=self.angle,stop_command=self.stop_time is not None,
                    stop_reason=self.reason,stop_time=self.stop_time,phase=self.phase,
                    initial_ml=len(p)*ml,capacity_geometric_ml=self.cup_geometry['capacity_geometric_ml'],
                    cup_inventory_fraction=cup_ml/self.cup_geometry['legacy_brim_nominal_ml'],
                    source_outflow_ml_s=flow_ml_s, predicted_cup_ml=float(predicted),
                    outside_peak_z_m=float(p[outside,2].max()) if outside.any() else None)

    def complete(self,t):
        return self.stop_time is not None and t>=self.stop_time+2.3

    def summary(self, rows):
        final=rows[-1]
        expected=self.spec.get('expected_stop') or ('bottle_empty' if self.spec['initial_ml']<CAPACITY_ML else 'cup_near_full')
        gates=[]
        if final['stop_reason'] != expected: gates.append('wrong_stop_condition')
        if expected=='cup_near_full' and final['cup_fill_fraction']<.80: gates.append('underfilled_cup')
        if expected=='cup_near_full' and final['cup_inventory_fraction']<.80: gates.append('low_retained_volume')
        if expected=='bottle_empty' and final['bottle_ml']>max(.25,.01*final['initial_ml']): gates.append('residual_bottle')
        if final['spilled_ml']+final['airborne_ml']>.05*final['initial_ml']: gates.append('excess_outside')
        if min(r['bottle_ml'] for r in rows if r['t']<=.65)<.99*self.spec['initial_ml']: gates.append('upright_source_leak')
        return dict(eligible=not gates,gates=gates,bottle_result=final,expected_stop=expected,
                    attribution='scripted bottle pose and simulator-state stop teacher',grasp_verified=False)
