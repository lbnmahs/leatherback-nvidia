import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

SOCCERBALL_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/SoccerBall",
    spawn=sim_utils.SphereCfg(
        radius=0.11,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.43),  # approx soccer ball
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.11),
    ),
)