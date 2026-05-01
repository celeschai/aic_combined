# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the AIC task MDP.

Generic terminations (``time_out``, etc.) are re-exported from
``isaaclab.envs.mdp`` via ``mdp/__init__.py``; add task-specific terminations
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_collision_triggered(
    env: "ManagerBasedRLEnv",
    threshold_n: float = 1.0,
) -> torch.Tensor:
    """Terminate (failure) when any tracked arm/gripper link contact exceeds ``threshold_n``."""
    sensor = env.scene.sensors.get("robot_contact")
    if sensor is None or sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    force_norm = torch.norm(sensor.data.net_forces_w, dim=-1)
    return force_norm.max(dim=-1).values > threshold_n
