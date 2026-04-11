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


def straight_stairs_terrain(difficulty: float, cfg: StraightStairsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a straight stair corridor aligned with the positive x-axis.

    The terrain keeps a flat floor over the full cell and extrudes a centered stair corridor.
    Curriculum difficulty increases the number of steps while shrinking tread depth.
    """

    stair_center_y = cfg.size[1] * 0.5
    stair_width = min(cfg.stair_width, cfg.size[1] - 2.0 * cfg.side_buffer)
    ground_thickness = 0.02
    corridor_min_y = stair_center_y - stair_width * 0.5
    corridor_max_y = stair_center_y + stair_width * 0.5

    # Three curriculum bands:
    # 1) low steps and wide treads
    # 2) denser stairs with light height perturbation
    # 3) longer stairs with light tread perturbation
    if difficulty < (1.0 / 3.0):
        local = difficulty / (1.0 / 3.0)
        base_step_height = 0.06 + 0.02 * local
        base_step_depth = 0.48 - 0.04 * local
        num_steps = 4 + int(local > 0.5)
        height_jitter = 0.0
        depth_jitter = 0.0
    elif difficulty < (2.0 / 3.0):
        local = (difficulty - 1.0 / 3.0) / (1.0 / 3.0)
        base_step_height = 0.085 + 0.025 * local
        base_step_depth = 0.40 - 0.03 * local
        num_steps = 6
        height_jitter = 0.5 * cfg.height_jitter
        depth_jitter = 0.0
    else:
        local = (difficulty - 2.0 / 3.0) / (1.0 / 3.0)
        base_step_height = 0.12 + 0.04 * local
        base_step_depth = 0.34 - 0.04 * local
        num_steps = 7 + int(local > 0.5)
        height_jitter = cfg.height_jitter
        depth_jitter = cfg.depth_jitter

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
        step_depth = max(
            0.18,
            base_step_depth + depth_jitter * np.cos(2.0 * np.pi * (0.5 * difficulty + local_phase)),
        )

        center_x = start_x + step_depth * 0.5
        current_top_z += step_height
        step_box = trimesh.creation.box(
            (step_depth, stair_width, current_top_z),
            trimesh.transformations.translation_matrix((center_x, stair_center_y, current_top_z * 0.5)),
        )
        meshes_list.append(step_box)
        start_x += step_depth

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
            proportion=0.2,
        ),
        "simple_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "straight_stairs": StraightStairsTerrainCfg(
            proportion=0.6,
            stair_width=1.8,
            start_platform_length=1.6,
            top_platform_length=1.6,
            step_height_range=(0.06, 0.16),
            step_depth_range=(0.42, 0.30),
            num_steps_range=(4, 8),
            height_jitter=0.008,
            depth_jitter=0.02,
            side_buffer=0.8,
        ),
    },
)
