# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .stair_env_cfg import UnitreeB2StairEnvCfg
from .stair_terrain_cfg import UNITREE_B2_STAIR_TERRAINS_PLAY_CFG


@configclass
class UnitreeB2StairPlayEnvCfg(UnitreeB2StairEnvCfg):
    """Play-time stair configuration following the source-task fixed-difficulty pattern."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = UNITREE_B2_STAIR_TERRAINS_PLAY_CFG

        # Keep play focused on forward stair traversal for repeatable inspection.
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Play mode still parses reward terms, so remove zero-weight placeholders inherited from the rough base.
        self.disable_zero_weight_rewards()
