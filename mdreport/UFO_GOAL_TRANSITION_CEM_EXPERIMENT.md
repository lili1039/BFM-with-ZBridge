# Test-Time Z-Bridge：23DoF 的 Goal / Reward 切换实验

本报告对应 README 展示的两条**完整仿真序列**，以及同一 checkpoint、训练切片 ID 7（`dance1_subject1__clip007`）生成的 dance tracking 视频。全部素材使用 `ufo_fb_g1_23dof_4096_seed4728_stable`（102,498,304 transitions）。对照基线 BFM-zero 使用 `--transition-mode hard` 直接切换 `z`；CEM 在切换时从当前机器人状态出发，搜索持续 30 个策略步的球面 `z` 路径。两组对照都使用相同任务顺序、模型、初始状态和渲染帧率（50 FPS）。

## Goal reaching：7 个目标

目标索引 `13 11 12 11 13 4 5`，每 40 步切换一次，共 280 步、6 次切换、5.6 秒。使用 23DoF FB stable checkpoint；两个模式都完成整条序列。CEM 每次用 6 个并行 MJLab 环境评估候选，运行 2 轮、3 个内部控制点和 2 个附加潜空间方向；末步 23 关节平均绝对角误差阈值是 0.25 rad。

- [并排机器人视频](../assets/demos/goal/comparison.mp4) · [同步 z 球面与关节力矩视频](../assets/demos/goal/z_sphere.mp4)
- [切换指标图](../assets/demos/goal/transition_metrics.png) · [逐次 CSV](../assets/demos/goal/transition_metrics.csv) · [逐次 JSON](../assets/demos/goal/transition_metrics.json)

| 6 个 30 步切换窗口的聚合指标 | BFM-zero | CEM |
| --- | ---: | ---: |
| 关节速度 RMS (rad/s) | 2.120 | 1.324 |
| 关节力矩 RMS (Nm) | 9.530 | 8.645 |
| 全局峰值关节力矩 (Nm) | 117.431 | 58.742 |

CEM 在全部 6 个窗口都降低了**窗口峰值**关节力矩，但窗口力矩 RMS 有 2 次高于直接切换。表中的 RMS 是把所有切换窗口的关节和时间样本合并后计算，不是 6 个窗口 RMS 的算术平均。

## Reward inference：6 个任务

任务顺序为慢速前进 → 侧移 → 原地旋转 → 后退 → 站立 → 低姿态站立；每 80 步切换一次，共 480 步、5 次切换、9.6 秒。使用 stable 23DoF FB checkpoint 及从**同一 checkpoint** 推断出的缓存 reward latent；两个模式都完成整条序列。CEM 的候选数、轮数、控制点和路径时长与 goal 实验相同。

- [并排机器人视频](../assets/demos/reward/comparison.mp4) · [同步 z 球面与关节力矩视频](../assets/demos/reward/z_sphere.mp4) · [六任务画面](../assets/demos/reward/task_montage.png)
- [切换指标图](../assets/demos/reward/transition_metrics.png) · [逐次 CSV](../assets/demos/reward/transition_metrics.csv) · [逐次 JSON](../assets/demos/reward/transition_metrics.json)

| 5 个 30 步切换窗口的聚合指标 | BFM-zero | CEM |
| --- | ---: | ---: |
| 关节速度 RMS (rad/s) | 1.516 | 1.194 |
| 关节力矩 RMS (Nm) | 11.692 | 10.356 |
| 全局峰值关节力矩 (Nm) | 96.772 | 78.551 |

Reward 在第 240 步切换的峰值力矩反而高于基线；表格是这条序列的聚合结果，并不保证每次切换都改善。Reward CEM 没有固定目标关节角，以候选 rollout 未失败作为可行性条件，不检查最终任务 reward 是否达成。

## 实现与复现

规划器在 [`goal_transition.py`](../humanoidverse/goal_transition.py)。候选路径首尾固定为两个任务 `z`，中间由单调进度曲线及低维偏移构成，每步重新投影到模型的 `z` 球面。规划从当前根状态、关节状态、动作及 actor 观测历史复制出发；候选 rollout 在 GPU 上并行。成本由关节速度 RMS、峰值速度、累计关节角行程和峰值关节力矩组成；goal 还以末步关节角误差约束可行性。规划过程中不改动模型权重。

视频中的三维球面是 256 维 `z` 的有损投影；下方曲线为**实际 rollout** 每一步 23 个关节的力矩 RMS，阴影标出切换窗口。统计图使用 trace 的实际值，不使用 CEM 的预测值。合成器为 [`compare_latent_transitions.py`](../humanoidverse/tools/compare_latent_transitions.py)。Goal、reward、tracking 的完整命令及 hard/CEM 视频合成步骤见[根目录 README](../README.md)。

这些结果仅在当前仿真、任务顺序和 checkpoint 上成立。CEM 每次切换都要做多个候选 rollout，规划耗时数秒；未经时延与硬件约束验证，不能直接用于实时电机控制。
