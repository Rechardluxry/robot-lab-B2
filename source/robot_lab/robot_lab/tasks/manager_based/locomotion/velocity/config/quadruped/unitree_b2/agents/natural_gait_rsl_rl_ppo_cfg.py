# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2RoughPPORunnerCfg,
)


@configclass
class UnitreeB2NaturalGaitPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    """PPO runner config for natural-gait fine-tuning.

    Resumes from model_21000.pt with a lower learning rate and reduced
    exploration noise to avoid disrupting the existing rough-terrain policy.
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_natural_gait"
        self.max_iterations = 30000

        # Resume from the rough terrain checkpoint
        self.resume = True
        self.load_run = "2026-04-08_17-38-16"
        self.load_checkpoint = "model_21000.pt"

        # Lower LR for fine-tuning (half of original 1e-3)
        self.algorithm.learning_rate = 5e-4

        # Reduce exploration noise — policy is already well-initialised
        self.policy.init_noise_std = 0.8
        self.algorithm.entropy_coef = 0.005
