# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import (
    UnitreeB2RoughEnvCfg,
)


@configclass
class UnitreeB2NaturalGaitEnvCfg(UnitreeB2RoughEnvCfg):
    """Fine-tuning config for natural trot gait.

    Builds on the rough terrain policy (model_21000.pt) and adds:
      - GaitReward to enforce FL-RR / FR-RL diagonal trot sync
      - feet_air_time to encourage proper swing phase
      - feet_air_time_variance to regularise step timing across legs
      - feet_slide / feet_stumble penalties for cleaner contact
      - feet_height positive reward for foot clearance
      - Strengthened symmetry (joint_mirror, action_mirror)
      - Tighter smoothness penalties (action_rate, joint_acc, joint_power)
      - feet_distance_y_exp to maintain natural stance width
    """

    def __post_init__(self):
        # Inherit all B2 rough settings (robot, terrain curriculum, observations, etc.)
        super().__post_init__()

        # ------------------------------------------------------------------
        # Gait enforcement — diagonal trot (FL-RR / FR-RL)
        # ------------------------------------------------------------------
        self.rewards.feet_gait.weight = 1.0
        self.rewards.feet_gait.params["std"] = math.sqrt(0.25)  # tighter sync kernel
        self.rewards.feet_gait.params["max_err"] = 0.2
        self.rewards.feet_gait.params["velocity_threshold"] = 0.3
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )

        # ------------------------------------------------------------------
        # Swing phase duration
        # ------------------------------------------------------------------
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.feet_air_time.params["threshold"] = 0.4  # natural B2 trot ~0.4 s swing
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Penalise unequal air times across legs (promotes regularity)
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ------------------------------------------------------------------
        # Contact quality
        # ------------------------------------------------------------------
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ------------------------------------------------------------------
        # Foot clearance during swing
        # ------------------------------------------------------------------
        self.rewards.feet_height.weight = 0.2
        self.rewards.feet_height.params["target_height"] = 0.06  # 6 cm clearance
        self.rewards.feet_height.params["tanh_mult"] = 2.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]

        # ------------------------------------------------------------------
        # Symmetry
        # ------------------------------------------------------------------
        self.rewards.joint_mirror.weight = -0.1  # doubled from rough's -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_mirror.weight = -0.05

        # ------------------------------------------------------------------
        # Smoothness / energy
        # ------------------------------------------------------------------
        self.rewards.action_rate_l2.weight = -0.02      # doubled from -0.01
        self.rewards.joint_acc_l2.weight = -2e-7        # doubled from -1e-7
        self.rewards.joint_power.weight = -2e-5         # doubled from -1e-5

        # ------------------------------------------------------------------
        # Stance geometry — maintain natural B2 stance width
        # ------------------------------------------------------------------
        self.rewards.feet_distance_y_exp.weight = 0.3
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.32
        self.rewards.feet_distance_y_exp.params["std"] = 0.1
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]

        # ------------------------------------------------------------------
        # Disable any remaining zero-weight rewards
        # ------------------------------------------------------------------
        self.disable_zero_weight_rewards()
