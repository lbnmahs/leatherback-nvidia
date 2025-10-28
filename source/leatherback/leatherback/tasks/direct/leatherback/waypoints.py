"""Waypoint visualization configuration for the Leatherback environment."""
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

# Visual markers for waypoint course: red=current target, green=future waypoints
WAYPOINT_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker0": sim_utils.SphereCfg(  # Current target (red)
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker1": sim_utils.SphereCfg(  # Future targets (green)
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)
