# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the AIC task — generic Cartesian-control building blocks."""

from isaaclab.envs.mdp import (
    UniformPoseCommandCfg,
    action_rate_l2,
    body_pose_w,
    generated_commands,
    image,
    joint_pos_rel,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    reset_joints_by_offset,
    reset_joints_by_scale,
    time_out,
)
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .events import (  # noqa: F401
    ARM_HOME_JOINT_LIST,
    ARM_HOME_JOINT_POS,
    randomize_dome_light,
)
from .observations import (  # noqa: F401
    contact_net_forces,
    ee_ang_vel_base,
    ee_lin_vel_base,
    ee_pos_base,
    ee_rpy_base,
    ee_to_target_ori_error,
    ee_to_target_pos_error,
)
from .rewards import (  # noqa: F401
    arm_collision_penalty,
    body_lin_acc_l2,
    ee_jerk_penalty,
    ee_velocity_penalty,
    joint_acc_l2,
    joint_pos_limits,
    joint_torques_l2,
    orientation_command_error,
    orientation_command_error_tanh,
    position_command_error,
    position_command_error_tanh,
)
from .terminations import arm_collision_triggered  # noqa: F401
