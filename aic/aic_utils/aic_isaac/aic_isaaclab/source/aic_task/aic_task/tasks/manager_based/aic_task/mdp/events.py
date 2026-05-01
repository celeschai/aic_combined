# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms (resets / randomization) for the AIC task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.usd
import torch
from pxr import Gf, UsdLux

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# UR5e home pose used by AIC scenes (matches Gazebo aic_controller home).
_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ARM_HOME_JOINT_POS: dict[str, float] = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}

ARM_HOME_JOINT_LIST: list[float] = [ARM_HOME_JOINT_POS[name] for name in _ARM_JOINT_NAMES]


def randomize_dome_light(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1500.0, 3500.0),
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ),
):
    """Randomize the global dome light intensity and color at reset.

    Note: a single dome light is shared across envs, so this affects all
    environments at once.
    """
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")
    if not light_prim.IsValid():
        return

    light = UsdLux.DomeLight(light_prim)
    intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
    light.GetIntensityAttr().Set(intensity)

    color_min, color_max = color_range
    r = torch.empty(1).uniform_(color_min[0], color_max[0]).item()
    g = torch.empty(1).uniform_(color_min[1], color_max[1]).item()
    b = torch.empty(1).uniform_(color_min[2], color_max[2]).item()
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))
