# 已完成

- [x] TASK-000: 项目基础设施初始化完成
- [x] TASK-001: 执行文档中的 smoke 命令 `python scripts/tools/list_envs.py`，并记录当前环境下精确的 exit 127 结果
- [x] TASK-003: 确认 README smoke 流程应使用的 conda 环境 `robotlab232_lxr`，其解释器路径为 `/home/hzzz/.conda/envs/robotlab232_lxr/bin/python`
- [x] TASK-004: 为 Unitree B2 新增盲走楼梯微调任务 `RobotLab-Isaac-Velocity-Stair-Unitree-B2-v0`，包含楼梯地形、任务注册、楼梯奖励与楼梯终止项，并保持 warm-start 兼容
- [x] TASK-005: 修复楼梯地形生成器的运行时阻塞，并在完整 Isaac Lab shell 中通过环境创建、reset 和一次零动作 step 验证 `RobotLab-Isaac-Velocity-Stair-Unitree-B2-v0`
- [x] TASK-006: 将 `memory/` 目录中当前保留的英文状态内容翻译为中文，并保持命令、路径、任务编号与文件名不变
- [x] TASK-007: 将 B2 stair 任务收敛为标准直楼梯前向通过的 stair v1，只保留 flat+straight stairs、固定 reset 朝向，并将命令范围压缩到前向小中速
- [x] TASK-008: 对 B2 stair v1 做稳定性修正，放松 body_collision termination、进一步放慢第一档楼梯 curriculum，并降低 PPO 探索强度以减少第一阶重启和步态退化
- [x] TASK-009: 按用户要求删除已不再继续的 TASK-002，并同步清理待办列表与交接状态
- [x] TASK-010: 为稳定版 B2 stair v1 增加边缘安全与顶部完成逻辑，加入顶部 success termination、中线耦合进度奖励、edge_proximity_penalty 和更居中的 reset
- [x] TASK-011: 确认当前仓库状态已同步到 GitHub，并修复 memory 状态与最新 Git 历史之间的记录差异
