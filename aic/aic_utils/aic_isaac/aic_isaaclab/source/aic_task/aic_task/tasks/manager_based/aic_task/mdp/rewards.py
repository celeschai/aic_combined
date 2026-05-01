# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic reward terms for Cartesian-controlled manipulation tasks.

Only task-agnostic shaping is included here: command-error tracking, joint /
body regularization, and a generic arm-collision penalty. Task-specific rewards
(insertion depth, contact bonuses, trajectory following, FT-force shaping, ...)
should be added by downstream task configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Command tracking
# ---------------------------------------------------------------------------

def position_command_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 position error between body and pose command (world frame)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=-1)


def position_command_error_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """tanh-shaped position tracking reward (1 at zero error, -> 0 far from goal)."""
    distance = position_command_error(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def orientation_command_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Quaternion-error magnitude (radians) between body orientation and pose command."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    _, des_quat_w = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], q=des_quat_b
    )
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """tanh-shaped orientation tracking reward (1 at zero error, -> 0 far from goal)."""
    err = orientation_command_error(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(err / std)


# ---------------------------------------------------------------------------
# Joint regularization
# ---------------------------------------------------------------------------

def joint_torques_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_acc_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def joint_pos_limits(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    out_of_limits = -(asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ---------------------------------------------------------------------------
# End-effector regularization
# ---------------------------------------------------------------------------

def body_lin_acc_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids]), dim=(1, 2))


def ee_velocity_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_tcp"]),
) -> torch.Tensor:
    """L2 norm of EE linear velocity (world frame)."""
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    lin_vel = asset.data.body_vel_w[:, body_id, :3]
    return torch.sum(torch.square(lin_vel), dim=-1)


def ee_jerk_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Squared change in EE linear velocity between consecutive RL steps.

    Requires the env to maintain ``_prev_ee_vel_w`` and ``_ee_body_idx``
    (provided by ``AICTaskEnv``).
    """
    if not hasattr(env, "_prev_ee_vel_w"):
        return torch.zeros(env.num_envs, device=env.device)
    ee_vel = env.scene["robot"].data.body_vel_w[:, env._ee_body_idx, :3]
    return torch.sum(torch.square(ee_vel - env._prev_ee_vel_w), dim=-1)


# ---------------------------------------------------------------------------
# Contact / safety
# ---------------------------------------------------------------------------

def arm_collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact"),
    threshold_n: float = 1.0,
) -> torch.Tensor:
    """Per-step indicator (0/1) when any tracked arm/gripper link contact exceeds ``threshold_n``.

    Apply via a negative ``weight`` in ``RewardsCfg`` to convert into a penalty.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_norm = torch.norm(sensor.data.net_forces_w, dim=-1)
    return (force_norm.max(dim=-1).values > threshold_n).float()
