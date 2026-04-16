# Unitree B2 Stair Migration

## Goal

This note records how the blind stairs curriculum from `IsaacLab-Quadruped-Tasks` maps into the
`robot_lab` manager-based locomotion stack for `Unitree B2`.

## Source-To-Target Mapping

- Source terrain generator:
  `/home/hzzz/lxr/IsaacLab-Quadruped-Tasks/exts/omni.isaac.lab_quadruped_tasks/omni/isaac/lab_quadruped_tasks/cfg/quadruped_terrains_cfg.py`
  maps to
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/stair_terrain_cfg.py`
- Source base env:
  `/home/hzzz/lxr/IsaacLab-Quadruped-Tasks/exts/omni.isaac.lab_quadruped_tasks/omni/isaac/lab_quadruped_tasks/cfg/quadruped_env_cfg.py`
  maps to
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Source robot-specific blind stairs env:
  `/home/hzzz/lxr/IsaacLab-Quadruped-Tasks/exts/omni.isaac.lab_quadruped_tasks/omni/isaac/lab_quadruped_tasks/robots/go2/go2_env_cfg.py`
  maps to
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/stair_env_cfg.py`
  and
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/stair_play_env_cfg.py`
- Source reward / curriculum / termination helpers:
  `/home/hzzz/lxr/IsaacLab-Quadruped-Tasks/exts/omni.isaac.lab_quadruped_tasks/omni/isaac/lab_quadruped_tasks/mdp/`
  map to
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/`
- Source task registration:
  `.../robots/go2/__init__.py`
  maps to
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`

## Terrain Curriculum

- Source stairs curriculum uses `TerrainGeneratorCfg(curriculum=True, difficulty_range=(0.0, 1.0))`
  with robot-specific envs switching `scene.terrain.terrain_generator` to a stairs preset.
- `robot_lab` mirrors the same hook by assigning `UNITREE_B2_STAIR_TERRAINS_CFG` in
  `UnitreeB2StairEnvCfg.__post_init__()`.
- The source task uses pyramid stairs and sloped transitions. For B2 stair-v1, the migrated target
  keeps a single straight-stairs corridor because it is easier to diagnose and preserves warm-start
  compatibility with the rough policy.
- `StraightStairsTerrainCfg` keeps stair geometry configurable:
  `stair_width`, `start_platform_length`, `top_platform_length`, `step_height_range`,
  `step_depth_range`, `num_steps_range`, `height_jitter`, and `depth_jitter`.
- `UNITREE_B2_STAIR_TERRAINS_PLAY_CFG` follows the source play pattern:
  fixed difficulty, curriculum disabled, single stairs terrain enabled.

## Environment Integration

- Observations and actions stay inherited from the B2 rough task to preserve the actor/critic I/O
  dimensions and the 12-dim joint-position action space.
- Stair-specific training changes live in `UnitreeB2StairEnvCfg`:
  forward-only command ranges, centered reset, stair rewards, stair terminations, and easy-row start
  via `scene.terrain.max_init_terrain_level = 0`.
- Stair play changes live in `UnitreeB2StairPlayEnvCfg`:
  fixed-difficulty terrain, no terrain-level curriculum, and zero yaw command for deterministic replay.

## Reward / Termination Hooks

- Reward config:
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/stair_env_cfg.py`
- Reward functions:
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- Termination config:
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/stair_env_cfg.py`
- Termination functions:
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`

The migrated task keeps the existing stair-specific logic:
- `stair_progress`
- `centerline_reward`
- `edge_proximity_penalty`
- `stall_penalty`
- `back_slip_penalty`
- `foot_clearance_reward`
- `stair_top_platform_success`
- `no_forward_progress`
- `continuous_back_slip`

## Runner And Registration

- PPO runner:
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py`
- Gym registrations:
  `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`

Registered task ids:
- `RobotLab-Isaac-Velocity-Stair-Unitree-B2-v0`
- `RobotLab-Isaac-Velocity-Stair-Play-Unitree-B2-v0`

## Warm-Start Workflow

The migration preserves warm-start compatibility with
`/home/hzzz/lxr/robot_lab/logs/rsl_rl/unitree_b2_rough/2026-04-08_17-38-16/model_21000.pt`
because the observation dimensions, action dimensions, and PPO network structure are unchanged.

## Commands

Train:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate robotlab232_lxr
env TERM=xterm bash ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stair-Unitree-B2-v0 \
  --headless \
  --resume \
  --load_run 2026-04-08_17-38-16 \
  --checkpoint model_21000.pt
```

Play:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate robotlab232_lxr
env TERM=xterm bash ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Stair-Play-Unitree-B2-v0 \
  --num_envs 1 \
  --checkpoint /home/hzzz/lxr/robot_lab/logs/rsl_rl/unitree_b2_rough/2026-04-08_17-38-16/model_21000.pt
```

## Compatibility Notes

- `IsaacLab-Quadruped-Tasks` uses `MeshPyramidStairsTerrainCfg` and related built-ins; `robot_lab`
  uses a custom straight-stairs mesh function. Geometry-aware reward and termination code therefore
  assumes the stair corridor is aligned with local `+x`.
- `train.py` resumes from the task experiment directory. If the rough checkpoint is reused for stair
  fine-tuning, keep `model_21000.pt` under the `unitree_b2_stair/<load_run>/` directory or link it there.
- `scene.terrain.max_init_terrain_level = 0` is intentional for training. The play task disables it
  to follow the source repo's fixed-difficulty play behavior.
