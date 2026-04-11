# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def no_forward_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    terrain_name: str,
    min_forward_speed: float,
    max_stall_duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when commanded stair climbing stalls for too long."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, "_stair_no_progress_steps"):
        env._stair_no_progress_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._stair_no_progress_steps[env.episode_length_buf <= 1] = 0

    on_stairs = is_robot_on_terrain(env, terrain_name)
    forward_cmd = env.command_manager.get_command(command_name)[:, 0]
    stalled = on_stairs & (forward_cmd > 0.1) & (asset.data.root_lin_vel_w[:, 0] < min_forward_speed)

    env._stair_no_progress_steps = torch.where(
        stalled,
        env._stair_no_progress_steps + 1,
        torch.zeros_like(env._stair_no_progress_steps),
    )

    max_steps = max(1, int(max_stall_duration_s / env.step_dt))
    return env._stair_no_progress_steps >= max_steps


def continuous_back_slip(
    env: ManagerBasedRLEnv,
    command_name: str,
    terrain_name: str,
    slip_speed_threshold: float,
    max_back_slip_duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot continuously slides backward on stairs."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, "_stair_back_slip_steps"):
        env._stair_back_slip_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._stair_back_slip_steps[env.episode_length_buf <= 1] = 0

    on_stairs = is_robot_on_terrain(env, terrain_name)
    forward_cmd = env.command_manager.get_command(command_name)[:, 0]
    back_slipping = on_stairs & (forward_cmd > 0.1) & (asset.data.root_lin_vel_w[:, 0] < -slip_speed_threshold)

    env._stair_back_slip_steps = torch.where(
        back_slipping,
        env._stair_back_slip_steps + 1,
        torch.zeros_like(env._stair_back_slip_steps),
    )

    max_steps = max(1, int(max_back_slip_duration_s / env.step_dt))
    return env._stair_back_slip_steps >= max_steps
