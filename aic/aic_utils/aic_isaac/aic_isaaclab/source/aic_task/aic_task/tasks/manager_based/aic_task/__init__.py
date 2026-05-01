# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AIC task — base env config, MDP modules, and Gym registration."""

import gymnasium as gym

gym.register(
    id="AIC-Task-v0",
    entry_point=f"{__name__}.aic_task_rl_env:AICTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_task_env_cfg:AICTaskEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_qual_ppo_cfg.yaml",
    },
)
