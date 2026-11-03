"""Uncalibrated comparison fixtures; original USD assets stay unchanged."""
SOURCE_SCALE = (.58, .58, .75)
SOURCE_POSITION = (.68, -.36, 1.065)
TARGET_POSITION = (.68, -.18, 1.065)
SOURCE_INNER_RADIUS = .046 * SOURCE_SCALE[0]
SOURCE_BOTTOM = -.070 * SOURCE_SCALE[2]
SOURCE_TOP = .080 * SOURCE_SCALE[2]
# PBD coefficients are not calibrated SI viscosities. Change one variable at a time.
LIQUIDS = (
 dict(id=1, cup='source_cup', label='Fine', color='Blue', rgb=(.05,.34,.74), spacing=.0028, viscosity=.01, position=SOURCE_POSITION),
 dict(id=2, cup='target_cup', label='Reference', color='White', rgb=(.88,.90,.92), spacing=.0036, viscosity=.01, position=TARGET_POSITION),
 dict(id=3, cup='cup_3', label='Coarse', color='Orange', rgb=(.95,.36,.06), spacing=.0044, viscosity=.01, position=(.68,0,1.065)),
 dict(id=4, cup='cup_4', label='Low viscosity', color='Cyan', rgb=(.03,.70,.70), spacing=.0036, viscosity=.001, position=(.68,.18,1.065)),
 dict(id=5, cup='cup_5', label='High viscosity', color='Purple', rgb=(.50,.12,.72), spacing=.0036, viscosity=100., position=(.68,.36,1.065)),
)
CUP_INTERIORS = tuple((c['cup'], SOURCE_INNER_RADIUS, SOURCE_BOTTOM, SOURCE_TOP) for c in LIQUIDS)
TABLE_BOUNDS = (.325, 1.175, -.525, .525)
NOMINAL_VOLUME_M3 = 62e-6
