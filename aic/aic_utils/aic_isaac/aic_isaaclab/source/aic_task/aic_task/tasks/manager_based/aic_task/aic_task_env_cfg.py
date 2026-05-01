# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic AIC task env config: scene + OSC infrastructure with placeholder MDP.

This module provides the *base* environment used by ``AICTaskEnv``:

- Scene: UR5e + gripper + cable, scene USD, task board.
- Action: 6-D Cartesian delta (raw joint efforts are overwritten by OSC).
- Observations: EE state in robot base frame + last action.
- Rewards: action-rate regularization only (placeholder).
- Terminations: time-out only.
- Events: dome-light randomization at reset.

Task-specific shaping (rewards, command terms, success detection, ...) should
live in a downstream config that subclasses ``AICTaskEnvCfg``.
"""

import os

import isaaclab.envs.mdp as il_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.events import ARM_HOME_JOINT_POS, randomize_dome_light

# Resolve asset directory relative to this file (portable across machines)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AIC_ASSET_DIR = os.path.join(_THIS_DIR, "Intrinsic_assets")
AIC_SCENE_DIR = AIC_ASSET_DIR
AIC_PARTS_DIR = os.path.join(AIC_ASSET_DIR, "assets")

# Rigid-body name regex for arm/gripper links (used by ContactSensor).
_AIC_ROBOT_CONTACT_BODIES_RE = (
    "(shoulder_link|upper_arm_link|forearm_link|wrist_1_link|wrist_2_link|wrist_3_link|"
    "flange|tool0|ati_base_link|ati_tool_link|gripper_hande_base_link|gripper_tcp|"
    "gripper_hande_finger_link_l|gripper_hande_finger_link_r|ft_frame)"
)

OSC_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

_EE_TCP = ["gripper_tcp"]


##
# Scene definition
##


@configclass
class AICTaskSceneCfg(InteractiveSceneCfg):
    """Scene for the AIC task: UR5e + gripper + cable, scene USD, task board."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/aic_unified_robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_ASSET_DIR, "aic_unified_robot_cable_sdf.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.2, 0.2, 1.14),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=dict(ARM_HOME_JOINT_POS),
        ),
        actuators={
            # Zero PD: joint torques come from AICTaskEnv OSC.
            "arm": IdealPDActuatorCfg(
                joint_names_expr=OSC_ARM_JOINTS,
                effort_limit_sim=150.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    aic_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/aic_scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_SCENE_DIR, "scene", "aic.usd"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    task_board = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/task_board",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                AIC_PARTS_DIR, "Task Board Base", "task_board_assembly.usd"
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, -0.2, 1.14), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Net contact forces (world frame) on tracked arm/gripper links.
    robot_contact = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/aic_unified_robot/aic_unified_robot/"
            + _AIC_ROBOT_CONTACT_BODIES_RE
        ),
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )


##
# MDP settings (placeholders)
##


@configclass
class CommandsCfg:
    """Generic Cartesian goal-pose command (uniform sampling around the EE).

    Subclass and replace ``ee_pose`` with a task-specific command for goal
    conditioning (e.g. an insertion target or a tracked frame).
    """

    ee_pose = il_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_tcp",
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=il_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.30, 0.50),
            pos_y=(-0.20, 0.20),
            pos_z=(1.10, 1.30),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-0.5, 0.5),
        ),
    )


@configclass
class ActionsCfg:
    """Dummy joint effort term — actual torques are written by ``AICTaskEnv._apply_osc``."""

    arm_action: ActionTerm = il_mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=OSC_ARM_JOINTS, scale=0.0
    )
    gripper_action: ActionTerm | None = None


@configclass
class EventCfg:
    """Reset-time events: small joint-position jitter + dome-light randomization."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )

    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 3500.0),
            "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        },
    )


@configclass
class TerminationsCfg:
    """Time-out only — add task-specific termination terms in subclasses."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ObservationsCfg:
    """Cartesian EE proprioception in robot-base frame, plus last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pos = ObsTerm(
            func=mdp.ee_pos_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_EE_TCP)},
        )
        ee_rpy = ObsTerm(
            func=mdp.ee_rpy_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_EE_TCP)},
        )
        ee_lin_vel = ObsTerm(
            func=mdp.ee_lin_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_EE_TCP)},
        )
        ee_ang_vel = ObsTerm(
            func=mdp.ee_ang_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_EE_TCP)},
        )
        actions = ObsTerm(func=il_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Placeholder rewards — only generic regularization. Task-specific shaping
    should be added by subclasses.
    """

    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-0.01)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    ee_jerk = RewTerm(func=mdp.ee_jerk_penalty, weight=-0.01)


##
# Environment configuration
##


@configclass
class AICTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Base AIC task env: scene + OSC infrastructure with placeholder MDP.

    Subclass and override ``commands`` / ``rewards`` / ``terminations`` /
    ``observations`` / ``events`` to define a concrete task.
    """

    # OperationalSpaceController gains (see ``AICTaskEnv._osc_setup``)
    osc_stiffness: list[float] = [300.0, 300.0, 300.0, 20.0, 20.0, 20.0]
    osc_damping: list[float] = [35.0, 35.0, 35.0, 9.0, 9.0, 9.0]
    osc_inertial_dynamics_decoupling: bool = False
    osc_gravity_compensation: bool = True
    osc_effort_limit: float = 150.0
    osc_ee_body: str = "gripper_tcp"

    # 6-D Cartesian delta scaling (raw action ∈ [-1, 1]).
    action_delta_pos_scale: float = 0.0005   # 0.5 mm/step → ~15 mm/s @ 30 Hz
    action_delta_ori_scale: float = 0.009    # ~0.5°/step  → ~15°/s   @ 30 Hz

    # IK reset (DLS) parameters — used by ``AICTaskEnv._ik_reset``.
    ik_steps: int = 80
    ik_lambda: float = 0.05
    ik_step_size: float = 0.5

    scene: AICTaskSceneCfg = AICTaskSceneCfg(num_envs=64, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 120.0
        self.scene.robot.actuators["arm"].effort_limit_sim = self.osc_effort_limit

        # Viewport / video framing.
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "gripper_tcp"
        self.viewer.env_index = 0
        self.viewer.eye = (0.22, 0.12, 0.20)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.scene.ground = None  # type: ignore[assignment]

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.08,
                    rot_sensitivity=0.05,
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(gripper_term=False, sim_device=self.sim.device),
                "spacemouse": Se3SpaceMouseCfg(gripper_term=False, sim_device=self.sim.device),
            },
        )
