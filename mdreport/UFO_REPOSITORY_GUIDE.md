# UFO 仓库导览：主要文件分别负责什么

本文对应当前 23DoF/CEM 代码。重点是 MJLab 中的 UFO-FB / G1 23DoF 路径及其通用工具。命令、训练参数、可视化和导出步骤见同目录的 [UFO_G1_23DOF_TUTORIAL.md](UFO_G1_23DOF_TUTORIAL.md)。本文件按职责解释文件组，不逐行解读。

## 从数据到模型的主线

```text
LaFAN 原始数据 → tools/retarget_g1_29dof_to_23dof.py → 23DoF NPZ
               → configs/data/*.yaml + utils/motion_data → full/train motion cache
run_train.sh → humanoidverse/train.py → training/workspace.py
                                      ├─ agents/envs/humanoidverse_mjlab.py → MJLab + motion library
                                      ├─ agents/presets/fb.py → fb → fb_cpr → fb_cpr_aux
                                      ├─ agents/buffers + expert_motion_loader
                                      └─ agents/evaluations → CSV / W&B / checkpoint
checkpoint → tracking/goal/reward_inference.py → tools/export_wsl_bundle.py → WSL/MuJoCo 交付包
                           └─ goal_transition.py + tools/compare_latent_transitions.py → CEM 对照视频/指标
```

`BFM-Zero` 在本仓库主要是环境、观测和奖励的 Hydra 配置名称；可直接训练的 CLI agent 是 `fb` 或 `tech`（`tldr` 只是 TeCH 的兼容别名），不是 `--agent bfm-zero`。本仓库默认使用 23DoF 配置、数据 manifest 和目标帧定义。

## 根目录、配置和文档

| 路径 | 职责 |
|---|---|
| `run_train.sh` | 推荐的训练入口包装器；设置外置 cache、uv、TorchInductor/Triton/Warp 和 EGL 环境后执行 `humanoidverse.train`。 |
| `pyproject.toml`、`uv.lock` | Python 依赖和可重复安装的锁定版本，包含适配本机 GPU 的 PyTorch 来源。 |
| `README.md`、`LICENSE` | 23DoF 快速运行、可视化结果与许可证；详细操作见本目录 tutorial。 |
| `assets/demos/` | README 使用的 23DoF tracking、goal 和 reward 精选视频与指标图。 |
| `configs/robots/g1_23dof.yaml` | 当前 23DoF 的 XML 路径、控制关节顺序、初始角、PD/力矩和接触语义；训练、推理、导出都应一致。 |
| `humanoidverse/data/robots/g1_23dof/` | 23DoF 机器人 XML 及其所引用的 mesh。 |
| `configs/data/lafan_g1_23dof_ik.yaml` | 当前 23DoF motion manifest：从 `UFO_DATA_ROOT` 读取数据，自动缓存 40 条完整序列和 902 条训练切片。 |
| `configs/goals/lafan_g1_23dof_ik.json` | 当前 23DoF goal reaching 的 14 个 motion/frame 目标。 |

`humanoidverse/config/` 是训练环境的 Hydra 配置树：`exp/bfm_zero/bfm_zero.yaml` 组合任务；`obs/bfm_zero_obs.yaml` 定义观测及历史；`rewards/reward_bfm_zero.yaml` 定义辅助奖励；`domain_rand/domain_rand.yaml` 定义随机化；`robot/g1/`、`simulator/`、`terrain/`、`env/` 则分别提供机器人、MuJoCo、地形和环境基础参数。它和项目根目录的 `configs/robots/` 不同：前者负责环境组合，后者是跨训练/导出共享的机器人契约。

## `humanoidverse/` 训练核心

| 路径 | 职责 |
|---|---|
| `train.py` | 解析 CLI，选择 FB/TeCH、GPU、数据 manifest、robot config、训练规模及 W&B，创建 `TrainConfig` 并启动训练。 |
| `train_mjlab.py` | 兼容入口；新的训练命令应使用 `run_train.sh` 或 `python -m humanoidverse.train`。 |
| `training/workspace.py` | 训练调度中心：创建环境/agent/replay、恢复 checkpoint、采样、更新、定期评估、CSV/W&B、保存模型与 buffer。 |
| `distributed.py` | 多卡进程与同步支持；当前示例为单卡 23DoF。 |
| `agents/envs/humanoidverse_mjlab.py` | 关键 MJLab 适配层：创建 G1 仿真、动作/PD、观测、随机化、reset、奖励和并行环境接口。 |
| `agents/envs/expert_motion_loader.py` | 从动作库生成专家 observation trajectory，供表征/判别训练使用。 |
| `agents/presets/fb.py` | FB 网络、优化器、辅助奖励及更新节奏的预设；当前 FB 组合实际使用 `FBcprAuxAgent`。 |
| `agents/presets/tldr.py` | TeCH 预设；文件名沿用旧 TLDR 命名，不能复用 FB checkpoint。 |
| `agents/fb/` | 基础 Forward-Backward 表征和 actor/critic 更新。 |
| `agents/fb_cpr/`、`agents/fb_cpr_aux/` | 在 FB 上叠加 CPR 判别器/critic 及辅助奖励 critic；当前 23DoF FB 的 actor loss、FB loss、辅助指标来自此链。 |
| `agents/tldr_dist_aux/`、`agents/gcr_rl*/` | TeCH/GCR 相关模型与 agent，不是当前 FB 主链路。 |
| `agents/nn_models.py`、`nn_filters.py`、`normalizers.py`、`misc/` | 共享网络组件、输入过滤、归一化、日志和 latent buffer。 |
| `agents/buffers/trajectory.py`、`transition.py` | 在线 replay 和 transition 存储；checkpoint 中的 `buffers/` 供续训和 reward inference。 |
| `agents/evaluations/humanoidverse_mjlab.py` | 按动作做 tracking evaluation、失败动作优先级及参考/策略并排视频。 |

`humanoidverse/utils/motion_data/` 负责 RobotState CSV/NPZ/pkl 的 schema、读取、转换、裁剪及 manifest cache；`utils/motion_lib/` 负责 skeleton、FK、动作载入/采样/插值；`utils/robot_spec/` 从 MuJoCo/URDF 和 YAML 解释关节、执行器与训练参数；`utils/reference_observations.py` 与 `envs/motion_observations.py` 处理参考观测。若出现动作 ID、关节顺序或维度不一致，优先沿这几层追查。

## 推理、可视化和交付工具

| 路径 | 职责与输出 |
|---|---|
| `tracking_inference.py` | 对完整动作生成逐步 tracking latent；可做 expert/policy 并排 MuJoCo MP4，也可用 `--latents-only` 只导出 NPZ。 |
| `goal_inference.py`、`goal_transition.py` | 从 goal JSON 生成目标 latent；可直接切换或用 MJLab 并行 CEM 搜索连续 z 路径，录制视频和切换 trace。 |
| `reward_inference.py`、`mjlab_reward_relabel.py` | 从 replay 重算 reward 并生成任务 latent；可逐任务录制，也可直接切换或用 CEM 连续执行任务序列。 |
| `tools/compare_latent_transitions.py` | 从 hard/CEM 的 summary 和 trace 合成并排机器人视频、z 球面与关节力矩视频、逐次指标图和 CSV/JSON。 |
| `export/backward_encoder.py`、`tools/export_backward_encoder_onnx.py` | 将 backward encoder 导出/验证为 ONNX。 |
| `tools/export_wsl_bundle.py` | 对指定 G1 23DoF 完整 checkpoint 一次性生成 ONNX、safetensors、reward/goal/tracking NPZ、机器人资产、控制契约、README、哈希与 tar.gz；不改写源 checkpoint。 |
| `tools/stage_continuation.py` | 将完整 checkpoint 复制到独立续训目录，检查关键文件和 inode，避免继续训练覆盖原件。 |
| `tools/retarget_g1_29dof_to_23dof.py` | 29DoF→23DoF 的约束 IK retarget，输出 NPZ 和质量报告。 |
| `tools/data_inspect.py`、`data_build.py`、`robot_inspect.py`、`eval_*` | 数据/机器人检查、构建和关节误差诊断工具。 |
| `tests/` | 训练 CLI、MJLab、motion schema、观测历史、robot config、ONNX、推理与交付包的回归测试；`test_export_wsl_bundle.py` 特别覆盖隔离续训和导出输入校验。 |

## 仓库外数据与一次 run 的结构

大型数据位于仓库外。本文约定从仓库根目录使用 `../data/UFO/`，并将 `UFO_DATA_ROOT` 指向它：`lafan_g1_23dof_ik/npz/` 存放 retarget 数据，`cache/motion_data/` 存放动作 cache，`runs/` 存放训练结果，`exports/` 存放 WSL 交付包。23DoF XML 和 mesh 位于仓库内。

一个训练 run 通常含 `config.yaml/json`、`console.log`、`train_log.txt`、`humanoidverse_tracking_eval.csv`、`videos/eval_<step>/`，以及 `checkpoint/` 下的 `train_status.json`、模型 safetensors、optimizer 和 replay buffer。`train_log.txt` 反映日志窗口，`train_status.json` 反映**已持久化、可恢复**的 checkpoint；两者不一定是同一步。W&B 视频还有自己的媒体副本，删除本地重复渲染片前须确认它已复制。

本仓库存在两个容易混淆的动作编号空间：训练和 W&B eval 使用 **902 条切片**（例如 `7 165 390 465 620`），而独立 tracking inference 和 WSL bundle 默认使用 **40 条完整动作**（`2 9 17 22 28`）。交付包的 `model/control_contract.json` 则是本地 MuJoCo 的 joint/observation/PD 权威说明；它不是可以跳过的附属文件。

建议阅读顺序：先看同目录 tutorial 和 `configs/robots/g1_23dof.yaml`、motion manifest；再看 `train.py → training/workspace.py → agents/envs/humanoidverse_mjlab.py → agents/presets/fb.py`；要部署时看三个 inference 入口、`tools/export_wsl_bundle.py` 与交付包自己的 `README.md`。
