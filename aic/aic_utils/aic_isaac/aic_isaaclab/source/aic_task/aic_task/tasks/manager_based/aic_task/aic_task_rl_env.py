# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AIC task RL environment with Cartesian operational-space control.

Action space
------------
6-D Cartesian delta: ``[Δx, Δy, Δz, Δroll, Δpitch, Δyaw]``. Position deltas are
expressed in the EE local frame and rotated to world before accumulation.
Scaled by ``cfg.action_delta_pos_scale`` (default 0.5 mm) and
``cfg.action_delta_ori_scale`` (default ~0.5°).

Control loop (per RL step, decimation=4 → 30 Hz RL / 120 Hz physics)
--------------------------------------------------------------------
1. Intercept raw action → update EE target pose (delta accumulation).
2. For each physics sub-step:
   a. ``apply_action()``  — writes 0 joint effort (dummy action term).
   b. ``_apply_osc()``    — overwrites effort with operational-space torques.
   c. ``scene.write_data_to_sim()`` / ``sim.step()`` / ``scene.update()``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

from .aic_task_env_cfg import AICTaskEnvCfg


class AICTaskEnv(ManagerBasedRLEnv):
    """Manager-based RL env with Cartesian OSC control + DLS IK reset."""

    cfg: AICTaskEnvCfg

    def __init__(self, cfg: AICTaskEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._osc_setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _osc_setup(self) -> None:
        """Cache robot handles, build OSC, allocate per-episode state."""
        self._robot = self.scene["robot"]

        body_name = self.cfg.osc_ee_body
        try:
            self._ee_body_idx = self._robot.body_names.index(body_name)
        except ValueError:
            self._ee_body_idx = next(
                i for i, n in enumerate(self._robot.body_names) if body_name in n
            )

        _arm_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._arm_joint_ids = torch.tensor(
            [self._robot.joint_names.index(n) for n in _arm_joints],
            device=self.device,
            dtype=torch.long,
        )

        # PhysX Jacobian skips the base link for fixed-base robots.
        self._jacobi_body_idx = self._ee_body_idx - 1

        # ── OSC ───────────────────────────────────────────────────────────
        Kp = list(self.cfg.osc_stiffness)
        Kd = list(self.cfg.osc_damping)
        ratios = [
            kd / (2.0 * math.sqrt(kp)) if kp > 0 else 1.0
            for kp, kd in zip(Kp, Kd)
        ]
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
            inertial_dynamics_decoupling=self.cfg.osc_inertial_dynamics_decoupling,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=self.cfg.osc_gravity_compensation,
            impedance_mode="fixed",
            motion_stiffness_task=Kp,
            motion_damping_ratio_task=ratios,
            nullspace_control="none",
        )
        self._osc = OperationalSpaceController(osc_cfg, self.num_envs, self.device)

        self._identity_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._identity_pose[:, 3] = 1.0

        self._target_pos_w: torch.Tensor | None = None
        self._target_quat_w: torch.Tensor | None = None

        # Per-episode state for jerk reward etc.
        N = self.num_envs
        self._prev_ee_pos_w = torch.zeros(N, 3, device=self.device)
        self._prev_ee_vel_w = torch.zeros(N, 3, device=self.device)

        print(
            f"\n[AICTaskEnv] ee_body={body_name}({self._ee_body_idx})  "
            f"Kp={Kp}  Kd={Kd}  ratios={[f'{r:.3f}' for r in ratios]}"
        )
        print(
            f"[AICTaskEnv] inertial_decoupling={self.cfg.osc_inertial_dynamics_decoupling} "
            f"gravity_comp={self.cfg.osc_gravity_compensation}"
        )

    # ------------------------------------------------------------------
    # RL step
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """RL step with Cartesian operational-space low-level controller."""
        action = action.to(self.device)

        if self._target_pos_w is None:
            self._seed_target_from_ee()

        self._update_target(action)

        self.action_manager.process_action(action)
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._apply_osc()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # EE position history (for jerk / path-length penalties).
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :]
        self._last_step_displacement = (ee_pos - self._prev_ee_pos_w).norm(dim=-1)
        self._prev_ee_pos_w = ee_pos.clone()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # EE velocity history AFTER reward (jerk reward reads prev).
        ee_vel = self._robot.data.body_vel_w[:, self._ee_body_idx, :3]
        self._prev_ee_vel_w = ee_vel.clone()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    # ------------------------------------------------------------------
    # OSC application
    # ------------------------------------------------------------------

    def _apply_osc(self) -> None:
        robot = self._robot

        jacs = robot.root_physx_view.get_jacobians()
        J_full = jacs[:, self._jacobi_body_idx, :, :].to(self.device)
        J_arm = J_full[:, :, self._arm_joint_ids]

        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)
        ee_vel = robot.data.body_vel_w[:, self._ee_body_idx, :]

        mass_matrix = None
        if self.cfg.osc_inertial_dynamics_decoupling:
            M_full = robot.root_physx_view.get_generalized_mass_matrices()
            ids = self._arm_joint_ids.cpu()
            mass_matrix = M_full[:, ids][:, :, ids].to(self.device)

        gravity = None
        if self.cfg.osc_gravity_compensation:
            g_full = robot.root_physx_view.get_gravity_compensation_forces()
            ids = self._arm_joint_ids.cpu()
            gravity = g_full[:, ids].to(self.device)

        target_pose = torch.cat([self._target_pos_w, self._target_quat_w], dim=-1)

        self._osc.set_command(
            command=target_pose,
            current_ee_pose_b=ee_pose,
            current_task_frame_pose_b=self._identity_pose,
        )
        tau_arm = self._osc.compute(
            jacobian_b=J_arm,
            current_ee_pose_b=ee_pose,
            current_ee_vel_b=ee_vel,
            mass_matrix=mass_matrix,
            gravity=gravity,
        )
        tau_arm = tau_arm.clamp(-self.cfg.osc_effort_limit, self.cfg.osc_effort_limit)

        num_joints = robot.num_joints
        if num_joints == 6:
            robot.set_joint_effort_target(tau_arm)
        else:
            efforts = torch.zeros(
                self.num_envs, num_joints, device=tau_arm.device, dtype=tau_arm.dtype
            )
            efforts[:, self._arm_joint_ids] = tau_arm
            robot.set_joint_effort_target(efforts)

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------

    def _seed_target_from_ee(self) -> None:
        self._target_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :].clone()
        self._target_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx, :].clone()

    def _update_target(self, action: torch.Tensor) -> None:
        pos_scale = self.cfg.action_delta_pos_scale
        ori_scale = self.cfg.action_delta_ori_scale

        # delta_pos is expressed in EE local frame; rotate to world.
        delta_pos_ee = action[:, :3] * pos_scale
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx, :]
        delta_pos = quat_apply(ee_quat, delta_pos_ee)
        delta_rpy = action[:, 3:6] * ori_scale

        self._target_pos_w = self._target_pos_w + delta_pos

        dq = quat_from_euler_xyz(delta_rpy[:, 0], delta_rpy[:, 1], delta_rpy[:, 2])
        self._target_quat_w = quat_mul(dq, self._target_quat_w)
        self._target_quat_w = self._target_quat_w / self._target_quat_w.norm(
            dim=-1, keepdim=True
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset envs: events (joint jitter + light) → command resample → seed OSC target."""
        super()._reset_idx(env_ids)  # events + command resample

        if len(env_ids) == 0:
            return

        # Sync world-frame goal so debug markers are current after resample.
        self.command_manager.get_term("ee_pose").compute(0.0)

        if self._target_pos_w is None:
            self._seed_target_from_ee()

        # Seed OSC target from actual EE pose (after any reset perturbation).
        self._target_pos_w[env_ids] = self._robot.data.body_pos_w[
            env_ids, self._ee_body_idx, :
        ].clone()
        self._target_quat_w[env_ids] = self._robot.data.body_quat_w[
            env_ids, self._ee_body_idx, :
        ].clone()

        # Seed EE history.
        self._prev_ee_pos_w[env_ids] = self._robot.data.body_pos_w[
            env_ids, self._ee_body_idx, :
        ]
        self._prev_ee_vel_w[env_ids] = self._robot.data.body_vel_w[
            env_ids, self._ee_body_idx, :3
        ]

    def _ik_reset(
        self,
        env_ids: torch.Tensor,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        body_idx: int | None = None,
        jacobi_body_idx: int | None = None,
    ) -> None:
        """Damped-least-squares IK to drive ``body_idx`` to a target pose.

        Each iteration writes joint positions directly to the physics state and
        steps the sim once for an updated Jacobian and FK. Joint velocities are
        zeroed every iteration so the robot is at rest when IK converges.
        """
        n = len(env_ids)
        if n == 0:
            return

        b_idx = body_idx if body_idx is not None else self._ee_body_idx
        jb_idx = jacobi_body_idx if jacobi_body_idx is not None else self._jacobi_body_idx

        lam2 = self.cfg.ik_lambda ** 2
        I6 = torch.eye(6, device=self.device).unsqueeze(0).expand(n, 6, 6)

        for _ in range(self.cfg.ik_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            body_pos = self._robot.data.body_pos_w[env_ids, b_idx, :]
            body_quat = self._robot.data.body_quat_w[env_ids, b_idx, :]

            pos_err = target_pos_w - body_pos
            quat_err = quat_mul(target_quat_w, quat_conjugate(body_quat))
            ori_err = axis_angle_from_quat(quat_err)
            err = torch.cat([pos_err, ori_err], dim=-1)

            jacs = self._robot.root_physx_view.get_jacobians()
            J_full = jacs[:, jb_idx, :, :].to(self.device)
            J = J_full[env_ids][:, :, self._arm_joint_ids]

            JJT = J @ J.transpose(-1, -2) + lam2 * I6
            dq = (
                J.transpose(-1, -2)
                @ torch.linalg.solve(JJT, err.unsqueeze(-1))
            ).squeeze(-1)

            curr_q = self._robot.data.joint_pos[env_ids][:, self._arm_joint_ids]
            new_q = curr_q + dq * self.cfg.ik_step_size

            self._robot.write_joint_state_to_sim(
                position=new_q,
                velocity=torch.zeros_like(new_q),
                joint_ids=self._arm_joint_ids,
                env_ids=env_ids,
            )
