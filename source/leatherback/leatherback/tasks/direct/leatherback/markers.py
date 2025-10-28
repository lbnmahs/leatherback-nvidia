"""Robot marker visualization configuration (heading arrows)."""
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Arrow markers showing robot's current heading (cyan) and target heading (red)
ROBOT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/robotMarkers",
    markers={
        "forward": sim_utils.UsdFileCfg(  # Current heading (cyan)
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.3, 0.3, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0)
            ),
        ),
        "target": sim_utils.UsdFileCfg(  # Target heading (red)
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.3, 0.3, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            ),
        ),
    },
)
