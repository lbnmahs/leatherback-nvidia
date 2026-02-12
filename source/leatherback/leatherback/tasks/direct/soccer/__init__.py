# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from . import agents
from .leatherback_soccer_stage_1 import LeatherbackStage1SoccerEnv, LeatherbackStage1SoccerEnvCfg
from .leatherback_soccer_stage_2 import LeatherbackStage2AdversarialSoccerEnv, LeatherbackStage2AdversarialSoccerEnvCfg

##
# Register Gym environments.
##


gym.register(
    id="Leatherback-Stage1-Soccer-v0",
    entry_point=LeatherbackStage1SoccerEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackStage1SoccerEnvCfg,
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Leatherback-Stage2-Soccer-v0",
    entry_point=LeatherbackStage2AdversarialSoccerEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackStage2AdversarialSoccerEnvCfg,
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)