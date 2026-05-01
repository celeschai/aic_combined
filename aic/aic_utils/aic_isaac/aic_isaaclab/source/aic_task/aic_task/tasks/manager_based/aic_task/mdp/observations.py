# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic observation functions for Cartesian-controlled manipulation tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Contact sensor
# ---------------------------------------------------------------------------

def contact_net_forces(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Net contact forces (world frame) flattened for policy obs.

    Returns shape ``(num_envs, num_bodies * 3)``. Body selection is via
    ``sensor_cfg.body_ids`` if set, else by regex match against
    ``sensor_cfg.body_names``.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = contact_sensor.data.net_forces_w  # (N, B, 3)
    body_ids = sensor_cfg.body_ids
    if body_ids is None or body_ids == slice(None):
        if getattr(sensor_cfg, "body_names", None) is not None:
            names = (
                [sensor_cfg.body_names]
                if isinstance(sensor_cfg.body_names, str)
                else sensor_cfg.body_names
            )
            pattern = re.compile(names[0] if len(names) == 1 else "|".join(names))
            body_ids = [
                i for i, b in enumerate(contact_sensor.body_names) if pattern.search(b)
            ]
            if body_ids:
                net = net[:, body_ids, :]
    else:
        net = net[:, body_ids, :]
    return net.reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# EE-to-target error observations
# ---------------------------------------------------------------------------

def ee_to_target_pos_error(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """(target_pos - ee_pos) in world frame — 3-D position error."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    return des_pos_w - ee_pos_w


def ee_to_target_ori_error(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Axis-angle orientation error (target * ee^-1) — 3-D."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    q_err = quat_mul(des_quat_w, quat_conjugate(ee_quat_w))
    return axis_angle_from_quat(q_err)


# ---------------------------------------------------------------------------
# EE pose & velocity in robot base frame
# ---------------------------------------------------------------------------

def ee_pos_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE position relative to robot base frame — 3-D (x, y, z)."""
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    return quat_apply_inverse(
        asset.data.root_quat_w, ee_pos_w - asset.data.root_pos_w
    )


def ee_rpy_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE orientation as (roll, pitch, yaw) in robot base frame — 3-D (radians)."""
    asset = env.scene[asset_cfg.name]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    q_rel = quat_mul(quat_conjugate(asset.data.root_quat_w), ee_quat_w)
    roll, pitch, yaw = euler_xyz_from_quat(q_rel)
    return torch.stack([roll, pitch, yaw], dim=-1)


def ee_lin_vel_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE linear velocity in robot base frame — 3-D."""
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :3]
    return quat_apply_inverse(asset.data.root_quat_w, vel_w)


def ee_ang_vel_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE angular velocity in robot base frame — 3-D."""
    asset = env.scene[asset_cfg.name]
    omega_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], 3:]
    return quat_apply_inverse(asset.data.root_quat_w, omega_w)
