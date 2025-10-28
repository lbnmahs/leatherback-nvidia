"""Leatherback vehicle robot configuration for Isaac Lab."""
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Robot configuration: NVIDIA Leatherback vehicle (4-wheel drive car)
LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Leatherback/leatherback.usd",
        # Physics properties for rigid bodies
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,  # Maximum linear velocity [m/s]
            max_angular_velocity=1000.0,  # Maximum angular velocity [rad/s]
            max_depenetration_velocity=100.0,  # Maximum penetration correction speed
            enable_gyroscopic_forces=True,  # Enable gyroscopic effects for wheels
        ),
        # Joint/articulation solver properties
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Disable self-collision checks
            solver_position_iteration_count=4,  # Iterations for position constraints
            solver_velocity_iteration_count=0,  # No velocity constraint iterations
            sleep_threshold=0.005,  # Threshold for putting bodies to sleep
            stabilization_threshold=0.001,  # Position stabilization threshold
        ),
    ),
    # Initial spawn state
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # Initial position [x, y, z] in meters
        joint_pos={
            # Wheel joints (controlled by throttle actuator)
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            # Steering joints (controlled by steering actuator)
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
            # Suspension joints (must be within valid ranges)
            "Shock__Rear_Right": -0.03,  # Rear: [-0.050, -0.010]
            "Shock__Rear_Left": -0.03,
            "Shock__Front_Right": 0.03,  # Front: [0.010, 0.050]
            "Shock__Front_Left": 0.03,
        },
    ),
    # Actuators: throttle (velocity control) and steering (position control)
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],  # All wheel joints
            effort_limit=40000.0,  # Maximum torque [Nm]
            velocity_limit=100.0,  # Maximum angular velocity [rad/s]
            stiffness=0.0,  # No position stiffness (pure velocity control)
            damping=100000.0,  # High damping for velocity tracking
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],  # Front steering joints
            effort_limit=40000.0,  # Maximum torque [Nm]
            velocity_limit=100.0,  # Maximum angular velocity [rad/s]
            stiffness=1000.0,  # Position stiffness for steering control
            damping=0.0,  # No damping (stiffness-based control)
        ),
    },
)
