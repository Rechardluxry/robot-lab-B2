# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


def _lerp(start: float, end: float, ratio: float) -> float:
    """Linearly interpolate between two scalars."""

    return start + (end - start) * float(np.clip(ratio, 0.0, 1.0))


def straight_stairs_terrain(difficulty: float, cfg: StraightStairsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a straight stair corridor aligned with the positive x-axis.

    The terrain keeps only a centered stair corridor plus narrow side support slabs.
    Curriculum is intentionally staged for a stair-v1 task: easy rows start with short,
    low stairs and wide treads, while later rows increase height, length and tread variation.
    """

    stair_center_y = cfg.size[1] * 0.5
    stair_width = min(cfg.stair_width, cfg.size[1] - 2.0 * cfg.side_buffer)
    ground_thickness = 0.02
    corridor_min_y = stair_center_y - stair_width * 0.5
    corridor_max_y = stair_center_y + stair_width * 0.5

    step_height_range = cfg.step_height_range
    step_depth_range = cfg.step_depth_range
    num_steps_range = cfg.num_steps_range
    stage_profiles = (
        {
            "height_fraction": (0.0, 0.25),
            "depth_fraction": (0.0, 0.20),
            "step_fraction": (0.0, 0.20),
            "height_jitter": 0.0,
            "depth_jitter": 0.0,
        },
        {
            "height_fraction": (0.25, 0.65),
            "depth_fraction": (0.20, 0.60),
            "step_fraction": (0.20, 0.65),
            "height_jitter": 0.35 * cfg.height_jitter,
            "depth_jitter": 0.0,
        },
        {
            "height_fraction": (0.65, 1.0),
            "depth_fraction": (0.60, 1.0),
            "step_fraction": (0.65, 1.0),
            "height_jitter": cfg.height_jitter,
            "depth_jitter": cfg.depth_jitter,
        },
    )
    stage_index = min(int(np.clip(difficulty, 0.0, 0.9999) * len(stage_profiles)), len(stage_profiles) - 1)
    local = np.clip(difficulty * len(stage_profiles) - stage_index, 0.0, 1.0)
    stage_profile = stage_profiles[stage_index]

    step_height_fraction = _lerp(*stage_profile["height_fraction"], local)
    step_depth_fraction = _lerp(*stage_profile["depth_fraction"], local)
    step_count_fraction = _lerp(*stage_profile["step_fraction"], local)
    base_step_height = _lerp(step_height_range[0], step_height_range[1], step_height_fraction)
    base_step_depth = _lerp(step_depth_range[0], step_depth_range[1], step_depth_fraction)
    num_steps = int(round(_lerp(float(num_steps_range[0]), float(num_steps_range[1]), step_count_fraction)))
    num_steps = int(np.clip(num_steps, num_steps_range[0], num_steps_range[1]))
    height_jitter = stage_profile["height_jitter"]
    depth_jitter = stage_profile["depth_jitter"]

    meshes_list: list[trimesh.Trimesh] = []

    # Fill the side areas with thin support slabs instead of a full plane.
    # This avoids coplanar overlap between the base plane and stair meshes during collider generation.
    left_width = max(0.0, corridor_min_y)
    if left_width > 0.0:
        meshes_list.append(
            trimesh.creation.box(
                (cfg.size[0], left_width, ground_thickness),
                trimesh.transformations.translation_matrix(
                    (cfg.size[0] * 0.5, left_width * 0.5, -ground_thickness * 0.5)
                ),
            )
        )
    right_width = max(0.0, cfg.size[1] - corridor_max_y)
    if right_width > 0.0:
        meshes_list.append(
            trimesh.creation.box(
                (cfg.size[0], right_width, ground_thickness),
                trimesh.transformations.translation_matrix(
                    (cfg.size[0] * 0.5, corridor_max_y + right_width * 0.5, -ground_thickness * 0.5)
                ),
            )
        )

    start_platform_length = min(cfg.start_platform_length, cfg.size[0])
    stair_run_limit = max(0.0, cfg.size[0] - start_platform_length - cfg.top_platform_length)
    if start_platform_length > 0.0:
        meshes_list.append(
            trimesh.creation.box(
                (start_platform_length, stair_width, ground_thickness),
                trimesh.transformations.translation_matrix(
                    (start_platform_length * 0.5, stair_center_y, -ground_thickness * 0.5)
                ),
            )
        )

    start_x = start_platform_length
    current_top_z = 0.0

    for step_idx in range(num_steps):
        local_phase = (step_idx + 1) / max(num_steps, 1)
        step_height = base_step_height + height_jitter * np.sin(2.0 * np.pi * (difficulty + local_phase))
        step_depth = base_step_depth + depth_jitter * np.cos(2.0 * np.pi * (0.5 * difficulty + local_phase))
        remaining_steps = max(num_steps - step_idx, 1)
        max_step_depth = stair_run_limit / remaining_steps if stair_run_limit > 0.0 else base_step_depth
        step_depth = max(0.18, min(step_depth, max_step_depth))

        center_x = start_x + step_depth * 0.5
        current_top_z += step_height
        step_box = trimesh.creation.box(
            (step_depth, stair_width, current_top_z),
            trimesh.transformations.translation_matrix((center_x, stair_center_y, current_top_z * 0.5)),
        )
        meshes_list.append(step_box)
        start_x += step_depth
        stair_run_limit = max(0.0, stair_run_limit - step_depth)

    # Use the remaining run as a flat exit plateau while ensuring the stairs themselves
    # never consume the reserved minimum top landing length from the curriculum config.
    top_platform_length = max(0.0, cfg.size[0] - start_x)
    if top_platform_length > 0.0:
        top_landing = trimesh.creation.box(
            (top_platform_length, stair_width, current_top_z),
            trimesh.transformations.translation_matrix(
                (start_x + top_platform_length * 0.5, stair_center_y, current_top_z * 0.5)
            ),
        )
        meshes_list.append(top_landing)

    # Spawn a little before the first step and align the centerline with the stair corridor.
    origin = np.array((start_platform_length * 0.5, stair_center_y, 0.0))
    return meshes_list, origin


@configclass
class StraightStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a straight stairway aligned with the +x direction."""

    function = straight_stairs_terrain

    stair_width: float = MISSING
    """Usable stair width along the y-axis."""

    start_platform_length: float = MISSING
    """Flat landing before the first step."""

    top_platform_length: float = MISSING
    """Flat landing after the last step."""

    step_height_range: tuple[float, float] = MISSING
    """Minimum and maximum step height used by the curriculum."""

    step_depth_range: tuple[float, float] = MISSING
    """Minimum and maximum tread depth used by the curriculum."""

    num_steps_range: tuple[int, int] = MISSING
    """Minimum and maximum number of steps used by the curriculum."""

    height_jitter: float = 0.0
    """Deterministic per-step height perturbation amplitude."""

    depth_jitter: float = 0.0
    """Deterministic per-step depth perturbation amplitude."""

    side_buffer: float = 0.0
    """Half-width margin on both sides of the stair corridor."""


UNITREE_B2_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=3,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),
        "straight_stairs": StraightStairsTerrainCfg(
            proportion=0.85,
            stair_width=1.7,
            start_platform_length=1.8,
            top_platform_length=1.6,
            step_height_range=(0.05, 0.15),
            step_depth_range=(0.50, 0.32),
            num_steps_range=(4, 8),
            height_jitter=0.006,
            depth_jitter=0.015,
            side_buffer=0.85,
        ),
    },
)
