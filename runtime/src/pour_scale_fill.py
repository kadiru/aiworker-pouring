"""Offline, finite cup-seed filling. Never admitted to autonomous episodes.

The source is an authored vertical jet. Existing particles are never repositioned
or deleted; the native solver handles impact and settling. Resulting cup seeds
still require independent containment, surface-stability and camera review.
"""
import math
import numpy as np
from pour_vessel_geometry import cup_profile


class SeedFill:
    def __init__(self, spec, initial_count):
        if not spec.get('calibration_only') or spec.get('policy_evaluation'):
            raise ValueError('Seed filling is offline calibration only')
        if spec.get('calibration_hold_angle_deg') is not None:
            raise ValueError('Seed filling requires the parked arm and bottle')
        config = spec['calibration_fill']
        self.spacing = float(spec['spacing'])
        self.flow = float(config['flow_nominal_ml_s'])
        self.target = float(config['target_height_fraction'])
        self.start = float(config['start_s'])
        self.duration = float(config['max_feed_s'])
        self.settle = float(config['settle_s'])
        values = [self.spacing, self.flow, self.target, self.start, self.duration, self.settle]
        if not np.isfinite(values).all() or self.spacing != .0018:
            raise ValueError('Finite values and the fixed 1.8 mm particle resolution required')
        if not (2 <= self.flow <= 8 and .85 <= self.target <= .92 and
                .3 <= self.start <= 1 and 1 <= self.duration <= 15 and 3 <= self.settle <= 6):
            raise ValueError('Outside the bounded seed-fill protocol')
        if type(initial_count) is not int or not 0 < initial_count < 20000:
            raise ValueError('Invalid initial inventory')
        self.initial_count = initial_count
        n = math.ceil(.005 / self.spacing)
        self.cross = np.asarray([(x*self.spacing, y*self.spacing)
                                 for x in range(-n, n+1) for y in range(-n, n+1)
                                 if (x*x+y*y)*self.spacing**2 <= .005**2])
        self.period = len(self.cross)*self.spacing**3*1e6 / self.flow
        self.speed = self.spacing/self.period
        self.height = cup_profile(spec)[-1][0]+4*self.spacing
        self.origin = np.asarray(spec['cup_world'], dtype=float)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError('Invalid cup frame')
        self.layers = 0
        self.stop_time = None
        self.stop_reason = None
        self.last_t = -math.inf
        self.last_observation_t = -math.inf
        self.confirmed = 0

    def additions(self, t, observation):
        if not math.isfinite(t) or t < self.last_t:
            raise ValueError('Seed-fill clock must be finite and monotonic')
        self.last_t = t
        if observation is not None and observation['t'] > self.last_observation_t:
            level = observation['cup_fill_fraction']
            if not math.isfinite(level):
                raise ValueError('Invalid measured calibration level')
            self.last_observation_t = observation['t']
            self.confirmed = self.confirmed+1 if level >= self.target else 0
        if self.stop_time is None and (self.confirmed >= 2 or t >= self.start+self.duration):
            self.stop_time = t
            self.stop_reason = 'measured_target' if self.confirmed >= 2 else 'finite_feed_limit'
        if self.stop_time is not None or t < self.start:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        due = math.floor((t-self.start)/self.period)+1
        # Refuse delayed catch-up that could create a whole compressed column.
        if due-self.layers > 2:
            raise ValueError('Emitter missed its native substep; no catch-up burst')
        positions, velocities = [], []
        for i in range(self.layers, due):
            age = t-(self.start+i*self.period)
            z = self.height-self.speed*age-.5*9.81*age*age
            positions.append(np.column_stack((self.cross, np.full(len(self.cross), z)))+self.origin)
            velocities.append(np.tile([0., 0., -self.speed-9.81*age], (len(self.cross), 1)))
        self.layers = due
        if not positions:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        return np.concatenate(positions).astype(np.float32), np.concatenate(velocities).astype(np.float32)

    @property
    def emitted_count(self):
        return self.layers*len(self.cross)

    def complete(self, t):
        return self.stop_time is not None and t >= self.stop_time+self.settle

    def evidence(self):
        return dict(kind='authored finite calibration jet, not robot pouring',
                    initial_count=self.initial_count, emitted_count=self.emitted_count,
                    emitted_nominal_ml=self.emitted_count*self.spacing**3*1e6,
                    flow_nominal_ml_s=self.flow, stop_time=self.stop_time,
                    stop_reason=self.stop_reason, target_height_fraction=self.target,
                    nozzle_world_m=(self.origin+[0., 0., self.height]).tolist(),
                    particle_mass_kg=1000*self.spacing**3)


def emit_native(trial):
    """Append after authoritative readback, preserving identities and set mass."""
    import omni.physx
    from pxr import Sdf, Vt, UsdPhysics
    fill = trial['seed_fill']; bot = trial['bottle']; inst = trial['inst']
    previous_count = fill.initial_count+fill.emitted_count
    additions, velocities = fill.additions(trial['t'], bot.last_observation)
    if not len(additions):
        return
    omni.physx.get_physx_interface().update_transformations(False, True, True, False)
    p = np.asarray(inst.GetPositionsAttr().Get(), dtype=np.float32).copy()
    v = np.asarray(inst.GetVelocitiesAttr().Get(), dtype=np.float32).copy()
    if len(p) != previous_count or len(v) != previous_count or len(bot.points) != previous_count:
        raise RuntimeError('Seed-fill particle inventory changed outside the emitter')
    p = np.concatenate((p, additions)); v = np.concatenate((v, velocities))
    if len(p) > 50000 or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise RuntimeError('Invalid or excessive calibration inventory')
    with Sdf.ChangeBlock():
        inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p))
        inst.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(v))
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0]*len(p)))
        if trial['spec'].get('mass_mode') != 'density':
            UsdPhysics.MassAPI(inst.GetPrim()).GetMassAttr().Set(len(p)*1000*fill.spacing**3)
    # The fixed-inventory observer retains its conservation assertion using the
    # explicitly appended inventory. No particle state is written back by it.
    bot.points = np.concatenate((bot.points, additions))
    if bot.ever_table_particles is not None:
        bot.ever_table_particles = np.pad(bot.ever_table_particles, (0, len(additions)))
    trial['expected_count'] = len(p)
