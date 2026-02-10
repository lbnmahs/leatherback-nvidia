# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from leatherback.assets.robots.leatherback import LEATHERBACK_CFG
from .waypoints import WAYPOINT_CFG
from .markers import ROBOT_MARKER_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    """Configuration for Leatherback vehicle waypoint-following environment."""
    
    # Environment timing: decimation=4 means actions applied every 4 sim steps
    decimation = 4
    episode_length_s = 20.0  # Maximum episode duration [s]
    
    # Action space: [throttle, steering]
    action_space = 2
    # Observation space: [position_error, cos(heading_error), sin(heading_error),
    #                     lin_vel_x, lin_vel_y, ang_vel_z, throttle_state, steering_state]
    observation_space = 8
    state_space = 0  # No privileged state information
    
    # Simulation settings: 60 Hz physics, render every 4 steps (15 Hz)
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    
    # Robot and visualization configurations
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG  # Waypoint marker visualization
    robot_marker_cfg = ROBOT_MARKER_CFG  # Robot heading marker visualization

    # Joint names for action mapping
    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    # Scene configuration
    env_spacing = 32.0  # Distance between parallel environments [m]
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=True
    )
