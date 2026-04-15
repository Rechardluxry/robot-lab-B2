# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg, TerminationsCfg

from .rough_env_cfg import UnitreeB2RoughEnvCfg
from .stair_terrain_cfg import UNITREE_B2_STAIR_TERRAINS_CFG


@configclass
class UnitreeB2StairRewardsCfg(RewardsCfg):
    """Reward terms for the blind stair fine-tuning task."""

    stair_progress = RewTerm(
        func=mdp.stair_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "terrain_name": "straight_stairs",
            "max_speed": 1.0,
            "centerline_full_reward_distance": 0.22,
            "centerline_decay_sigma": 0.18,
        },
    )
    centerline_reward = RewTerm(
        func=mdp.centerline_reward,
        weight=0.0,
        params={"terrain_name": "straight_stairs", "sigma": 0.35},
    )
    stall_penalty = RewTerm(
        func=mdp.stall_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "terrain_name": "straight_stairs",
            "min_forward_speed": 0.12,
            "action_scale": 0.35,
        },
    )
    back_slip_penalty = RewTerm(
        func=mdp.back_slip_penalty,
        weight=0.0,
        params={"command_name": "base_velocity", "terrain_name": "straight_stairs", "slip_speed_threshold": 0.05},
    )
    foot_clearance_reward = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "terrain_name": "straight_stairs",
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "target_height": -0.18,
            "std": 0.10,
            "tanh_mult": 2.0,
        },
    )
    body_collision_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    edge_proximity_penalty = RewTerm(
        func=mdp.edge_proximity_penalty,
        weight=0.0,
        params={
            "terrain_name": "straight_stairs",
            "stair_width": 1.0,
            "safe_margin": 0.24,
            "power": 2.0,
        },
    )


@configclass
class UnitreeB2StairTerminationsCfg(TerminationsCfg):
    """Termination terms for the blind stair fine-tuning task."""

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.75, "asset_cfg": SceneEntityCfg("robot")},
    )
    no_forward_progress = DoneTerm(
        func=mdp.no_forward_progress,
        params={
            "command_name": "base_velocity",
            "terrain_name": "straight_stairs",
            "min_forward_speed": 0.08,
            "max_stall_duration_s": 2.0,
        },
    )
    continuous_back_slip = DoneTerm(
        func=mdp.continuous_back_slip,
        params={
            "command_name": "base_velocity",
            "terrain_name": "straight_stairs",
            "slip_speed_threshold": 0.08,
            "max_back_slip_duration_s": 0.8,
        },
    )
    body_collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 60.0},
    )
    success = DoneTerm(
        func=mdp.stair_top_platform_success,
        params={
            "terrain_name": "straight_stairs",
            "terrain_length": 8.0,
            "start_platform_length": 2.0,
            "top_platform_length": 1.8,
            "lateral_tolerance": 0.45,
            "pitch_roll_threshold": 0.45,
            "success_duration_s": 0.02,
            "entry_margin": 0.00,
            "far_edge_margin": 0.05,
        },
    )


@configclass
class UnitreeB2StairEnvCfg(UnitreeB2RoughEnvCfg):
    rewards: UnitreeB2StairRewardsCfg = UnitreeB2StairRewardsCfg()
    terminations: UnitreeB2StairTerminationsCfg = UnitreeB2StairTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = UNITREE_B2_STAIR_TERRAINS_CFG
        # Start every episode from the easiest stair row and let curriculum move upward.
        self.scene.terrain.max_init_terrain_level = 0
        stair_cfg = self.scene.terrain.terrain_generator.sub_terrains["straight_stairs"]
        terrain_length = self.scene.terrain.terrain_generator.size[0]

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.08),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.02),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.01, 0.01),
                "z": (-0.02, 0.02),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.02, 0.02),
            },
        }

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = 0.0

        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_power.weight = -1e-5
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.5e-4

        self.rewards.track_lin_vel_xy_exp.weight = 2.2
        self.rewards.track_ang_vel_z_exp.weight = 0.4
        self.rewards.joint_pos_limits.weight = -1.5
        self.rewards.stand_still.weight = -0.3
        self.rewards.joint_pos_penalty.weight = -0.3
        self.rewards.joint_mirror.weight = -0.02
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.upward.weight = 4.5

        self.rewards.stair_progress.weight = 2.5
        self.rewards.centerline_reward.weight = 1.2
        self.rewards.centerline_reward.params["sigma"] = 0.25
        self.rewards.stall_penalty.weight = -1.5
        self.rewards.back_slip_penalty.weight = -1.5
        self.rewards.foot_clearance_reward.weight = 0.6
        self.rewards.foot_clearance_reward.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.body_collision_penalty.weight = -3.0
        self.rewards.edge_proximity_penalty.weight = -1.0
        self.rewards.edge_proximity_penalty.params["stair_width"] = stair_cfg.stair_width
        self.rewards.body_collision_penalty.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*_hip",
            ".*_thigh",
            ".*_calf",
        ]

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        # B2 does not expose a separate torso link in the URDF, so base_link covers trunk collisions here.
        self.terminations.body_collision.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*_thigh",
        ]
        self.terminations.success.params["terrain_length"] = terrain_length
        self.terminations.success.params["start_platform_length"] = stair_cfg.start_platform_length
        self.terminations.success.params["top_platform_length"] = stair_cfg.top_platform_length

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.05, 0.05)
        self.commands.base_velocity.ranges.heading = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2StairEnvCfg":
            self.disable_zero_weight_rewards()
