# 已训练 UFO 模型的 Reward、Goal 与 Tracking 推理

本文按当前代码仓库的 FB 模型路径，说明如何用一个已训练的 checkpoint 生成 reward、goal 和 tracking 条件向量 `z`，以及如何用这些向量控制策略。代码入口分别是 [`reward_inference.py`](../humanoidverse/reward_inference.py)、[`goal_inference.py`](../humanoidverse/goal_inference.py) 和 [`tracking_inference.py`](../humanoidverse/tracking_inference.py)。

## 1. 三种模式的共同结构

三种脚本都从 `<model-folder>/checkpoint/model/` 加载模型，调用 `model.act(observation, z, mean=True)` 输出确定性动作。环境配置来自 `<model-folder>/config.json`，并可在推理时指定机器人配置和 motion 数据。模型的机器人、关节顺序、动作维度及观测结构必须与 checkpoint 匹配。入口可参见 [`load_model_from_checkpoint_dir()`](../humanoidverse/agents/load_utils.py) 和 [`load_mjlab_env_cfg()`](../humanoidverse/mjlab_inference_utils.py)。

| 模式 | `z` 的来源 | 更新频率 | 主要输出 |
| --- | --- | --- | --- |
| Reward inference | 给 replay buffer 中的状态按新 reward 打分，再汇总其 backward encoding | 单任务或 `independent` rollout 固定；连续任务切换可用 `hard/cem` | `reward_inference/reward_latents.npz` |
| Goal reaching | 参考 motion 中指定的一帧 | `hard` 直接切换；`cem` 在切换窗口执行逐步变化的 `z` 路径 | `goal_inference/goal_latents.npz` |
| Tracking | 参考 motion 的每一帧 | 每个策略步更换 | `tracking_inference/zs_<motion_id>.npz` |

FB 模型的 `backward_map` 先用 checkpoint 保存的 observation normalizer 处理输入，再通过 backward encoder 得到 `B(s)`。仓库当前 FB preset 的 backward encoder 使用 `state` 和 `privileged_state`；actor 使用 `state`、`last_action`、`history_actor` 和额外传入的 `z`。该 preset 的 `z_dim=256`、`norm_z=True`，因此 `project_z(z) = sqrt(256) * normalize(z)`；实际部署时应以目标 checkpoint 的模型配置为准。参见 [`fb.py`](../humanoidverse/agents/presets/fb.py) 和 [`FBModel`](../humanoidverse/agents/fb/model.py)。

参考 motion 的 `state` 由相对默认关节角、关节速度、机身坐标系下的重力方向、机身角速度依次拼接；`privileged_state` 是各身体位置、姿态、线速度及角速度组成的 `max_local_self_obs`。生成参考状态时，`last_action` 用零填充。运行策略时，actor 接收的是仿真环境当前的实际观测，所以即使 tracking 的 `z` 序列预先算好，控制仍是闭环的。参见 [`get_backward_observation()`](../humanoidverse/utils/helpers.py)。

## 2. Tracking：Motion 如何定义、从哪里来

### 2.1 数据格式与 motion ID

参考动作可通过 `--data-path` 直接指定 motion 文件，也可使用 `--data-manifest <yaml> --dataset <name>` 选择 manifest 中的数据集。manifest 支持原生 UFO motion `pkl`，以及需要转换的 RobotState `npz`、`csv`；后两种会生成 MotionLib 所需的 UFO `pkl` 缓存。格式入口在 [`adapters.py`](../humanoidverse/utils/motion_data/adapters.py) 和 [`manifest.py`](../humanoidverse/utils/motion_data/manifest.py)。

UFO motion 文件是按动作名组织的字典，每条 motion 至少需要：

- `root_trans_offset`：形状 `[T, 3]` 的根位置；
- `pose_aa`：形状 `[T, J, 3]` 的逐身体 axis-angle 姿态；
- `fps`：该 motion 的原始帧率。

RobotState 输入则提供逐帧 `root_pos`、`root_quat`、`dof_pos` 和 `fps`。转换器根据机器人配置检查关节数量与名称，必要时把 DOF 重排为控制关节顺序，再构造 UFO motion。参见 [`schema.py`](../humanoidverse/utils/motion_data/schema.py)、[`robot_state.py`](../humanoidverse/utils/motion_data/robot_state.py) 和 [`robot_state_convert.py`](../humanoidverse/utils/motion_data/robot_state_convert.py)。

23DoF 示例 [`configs/data/lafan_g1_23dof_ik.yaml`](../configs/data/lafan_g1_23dof_ik.yaml) 从 retarget 后的 NPZ 自动生成训练切片和推理用完整动作。`--motion-list` 的数字是**所选数据集内的 motion ID**，本质上是 MotionLib 加载动作字典后的索引，而不是文件名。训练切片与推理完整动作的 ID 可以不同；换数据集或重建缓存后，应重新核对索引与动作名。README 的跳舞视频选用训练切片 `dance1_subject1__clip007`（ID 7），因此复现时须将 `--data-path` 指向 `lafan_g1_23dof_ik_train_near10s_ufo.pkl`。MotionLib 的载入顺序见 [`motion_lib_base.py`](../humanoidverse/utils/motion_lib/motion_lib_base.py)。

要加入自己的 tracking motion，可沿用该 manifest：准备同一机器人上的逐帧 NPZ，每个文件至少含 `root_pos [T,3]`、`root_quat [T,4]`、`dof_pos [T,num_dof]`；帧率可放在 NPZ 的 `fps`、`time` 中，或在 manifest 写 `fps`。推荐放 `joint_names`，让读取器按机器人控制关节顺序重排；如果省略，读取器会假定 `dof_pos` 已经是该顺序。四元数顺序依 robot config 中的 `root_quat_order` 解释。把 manifest 的 `source_path` 改为这些 NPZ 的路径或通配符，用 `--data-manifest` 和 `--dataset` 运行即可；数据变化后可加 `--rebuild-motion-cache`。原始 fps 与策略步长可以不同，MotionLib 会在环境 `dt` 上取得参考状态。字段读取见 [`robot_state_readers.py`](../humanoidverse/utils/motion_data/robot_state_readers.py)。

### 2.2 参考轨迹如何转换为逐步 `z`

[`tracking_inference.py`](../humanoidverse/tracking_inference.py) 加载所选数据集的全部 motion。对每个指定的 motion，它按环境的 `dt` 从 MotionLib 取得参考身体状态，通过 `get_backward_observation()` 构造每个策略时刻的观测；帧数约为 `ceil(motion_length / env.dt)`。第 0 帧用于重置仿真机器人。从第 1 帧起逐帧编码：

```text
z[t] = project_z(backward_map(reference_observation[t + 1]))
action[t] = model.act(current_environment_observation[t], z[t], mean=True)
```

输出的 `zs_<motion_id>.npz` 有 `motion_id` 和形状为 `[策略步数, z_dim]` 的 `z`。`--latents-only` 只生成向量；`--save-mp4 true` 则生成参考与策略并排视频。参考侧由 robot XML 和参考 `qpos` 直接播放；策略侧显示仿真 rollout。参考根四元数写入 MuJoCo `qpos` 时转为 `wxyz`，关节也会按 MuJoCo `qpos` 地址重排。参见 [`_expert_qpos_from_obs()`](../humanoidverse/tracking_inference.py) 和 [`run_tracking_inference()`](../humanoidverse/tracking_inference.py)。

**实现细节：**模型类另有一个使用 `seq_length` 平滑若干帧的 `tracking_inference()` 方法，但当前 CLI 不调用它。CLI 使用自己的 `_tracking_z()`，其中 `z[step:step+1].mean()` 仅包含一帧，因此实际没有时间平滑。参见 [`_tracking_z()`](../humanoidverse/tracking_inference.py) 与 [`FBModel.tracking_inference()`](../humanoidverse/agents/fb/model.py)。

## 3. Goal reaching：指定目标帧并编码

goal 文件是 JSON 列表，每项有 `motion_id`、`frames` 和用于命名的 `motion_name`。23DoF 的例子在 [`configs/goals/lafan_g1_23dof_ik.json`](../configs/goals/lafan_g1_23dof_ik.json)：

```json
{
  "motion_id": 9,
  "frames": [2193, 2230],
  "motion_name": "fallAndGetUp1_subject4"
}
```

程序用 `motion_id` 选中推理数据集中的完整 motion，再按 `frames` 取采样后的参考帧。`motion_name` 用于输出名称，应与所指动作核对。当前编码循环真正读取的是 MotionLib 对应帧，而不是 JSON 中可能附带的关节数值；超出轨迹长度的 frame 会被跳过。目标观测由 `get_backward_observation(..., velocity_multiplier=0)` 构造，然后计算：

```text
z_goal = project_z(backward_map(goal_observation_at_frame))
```

这给出的是一个固定目标状态向量，不是一段逐帧参考轨迹。输出 `goal_latents.npz` 的 `names` 与 `latents` 一一对应。可选视频从第一个 demo 的第 0 帧重置，每隔 `--goal-switch-interval` 个策略步换一个目标。默认 `--transition-mode hard` 在切换时直接替换 `z`；`cem` 使用 [`goal_transition.py`](../humanoidverse/goal_transition.py) 从当前机器人状态采样并评估 `--transition-steps` 步路径，到达下一目标后保持新向量。参见 [`goal_inference.py`](../humanoidverse/goal_inference.py)。

23DoF G1 默认读取 `configs/goals/lafan_g1_23dof_ik.json`；自定义数据集或机器人应显式传 `--goal-json`。JSON 的 motion ID 必须匹配本次选用的推理数据集。

## 4. Reward inference：修改 reward 并转换为 `z`

### 4.1 Replay buffer 与离线重标注

Reward inference 从 checkpoint 的 replay buffer 取数据，查找顺序是 `checkpoint/buffers/train_reduced`、`train_rank_<buffer-rank>`、旧格式 `train`；也可用 `--buffer-path` 显式指定。它采样 transition 的下一时刻 `qpos/qvel`、对应 action 和下一时刻 observation。参见 [`_load_replay_buffer()`](../humanoidverse/reward_inference.py) 和 [`RewardWrapperHV.reward_inference()`](../humanoidverse/mjlab_reward_relabel.py)。

任务 reward 定义在 [`humanoidverse/envs/g1_env_helper/rewards.py`](../humanoidverse/envs/g1_env_helper/rewards.py)。每个任务类继承 `RewardFunction`：`reward_from_name()` 解析 `--tasks` 中的名称，`compute(model, data)` 根据 MuJoCo 状态计算标量分数。例如：

- `move-ego-0-0.3` 由 `LocomotionReward` 解析，结合骨盆高度、身体朝向、质心速度及目标方向评分；
- `rotate-z-5-0.5` 由 `RotationReward` 解析，按目标角速度、站立高度和姿态评分。

新建任务时，在该文件增加 `RewardFunction` 子类及名称解析，再将新名称传给 `--tasks`。[`make_reward_from_name()`](../humanoidverse/mjlab_reward_relabel.py) 会查找这些类；重标注器逐条设置缓存 `qpos/qvel/action`，执行 `mujoco.mj_forward()` 后计算 reward。修改 reward 不会改写训练好的模型权重。

当前 23DoF 入口接受 `move-ego-*`、`rotate-z-*` 等 locomotion 任务。要扩展任务，除 reward 本身外，还要修改 [`_resolve_reward_tasks()`](../humanoidverse/reward_inference.py)，并检查 reward 依赖的 body/sensor 名称和 robot XML。

### 4.2 Reward 如何成为固定任务向量

设 replay 中第 `i` 个下一时刻状态的编码为 `B(s'_i)`，新 reward 为 `r_i`。FB 实现计算：

```text
w_i      = softmax(10 * r)_i
z_raw    = Σ_i r_i * w_i * B(s'_i)
z_reward = project_z(z_raw)
```

对应代码是 [`FBModel.reward_wr_inference()`](../humanoidverse/agents/fb/model.py)。`--num-samples` 决定一次采样多少 transition，`--n-inferences` 可重复采样，产生同一任务的多个向量。`reward_latents.npz` 保存任务名和向量，`reward_locomotion.pkl` 保存可复用缓存。`--reward-latents-path` 可从**匹配同一 checkpoint**的缓存直接读取，跳过 replay 重标注。默认 `--transition-mode independent` 为每个任务独立重置并使用固定 `z`；`hard` 在同一条轨迹中直接切换任务，`cem` 为切换窗口搜索连续 `z` 路径。路径搜索在测试时运行，不更新策略权重，也不把此次分数作为训练信号。新的 reward 只有在 replay 中有相关状态覆盖时，才有机会给出有用的 `z`。

这里的加权汇总依赖 FB 模型的 `reward_wr_inference()`。仓库也有其他模型类型；应先确认目标 checkpoint 的模型类别及该方法的语义，再把这套 reward 公式用于它们。

## 5. 23DoF 示例命令

以下命令从仓库根目录执行，并使用当前演示所用的 stable checkpoint。外置数据默认放在仓库同级的 `../data/UFO/`；`UFO_MODEL` 指向包含 `checkpoint/` 和 `config.json` 的训练目录。命令展示**仅导出 latent** 的最短路径。Tracking 若要生成视频，需去掉 `--latents-only`，并加 `--save-mp4 true`；goal 和 reward 可直接启用 `--save-mp4 true`，reward 同时需关闭 `--skip-rollouts`。7-goal / 6-reward 长序列和视频合成命令见[根目录 README](../README.md)；已生成的视频见[实验报告](UFO_GOAL_TRANSITION_CEM_EXPERIMENT.md)。

```bash
export UFO_DATA_ROOT="${UFO_DATA_ROOT:-$(pwd)/../data/UFO}"
export UFO_MODEL="$UFO_DATA_ROOT/runs/ufo_fb_g1_23dof_4096_seed4728_stable"
```

Tracking，只导出逐步 latent：

```bash
uv run python -m humanoidverse.tracking_inference --model-folder "$UFO_MODEL" --data-manifest configs/data/lafan_g1_23dof_ik.yaml --dataset lafan_g1_23dof_ik --robot-config configs/robots/g1_23dof.yaml --motion-list 2 9 --latents-only --export-onnx false
```

Goal reaching，按 JSON 中的目标帧导出固定 latent：

```bash
uv run python -m humanoidverse.goal_inference --model-folder "$UFO_MODEL" --data-manifest configs/data/lafan_g1_23dof_ik.yaml --dataset lafan_g1_23dof_ik --robot-config configs/robots/g1_23dof.yaml --goal-json configs/goals/lafan_g1_23dof_ik.json --save-mp4 false
```

Reward inference，从 replay buffer 生成两个任务的固定 latent：

```bash
uv run python -m humanoidverse.reward_inference --model-folder "$UFO_MODEL" --data-manifest configs/data/lafan_g1_23dof_ik.yaml --dataset lafan_g1_23dof_ik --robot-config configs/robots/g1_23dof.yaml --tasks move-ego-0-0.3 rotate-z-5-0.5 --num-samples 150000 --skip-rollouts true
```

推理前优先核对 checkpoint 的机器人配置、motion 的关节顺序，以及 goal JSON 中的 motion ID。推理数据可以是完整动作，训练数据可能是切片；两者的 motion ID 不能直接混用。要比较切换效果，应在同一个 checkpoint、任务顺序、初始状态和输出目录下分别运行 `hard` / `cem`；reward 的连续序列模式需要显式设置 `--transition-mode hard` 或 `cem`，否则默认逐任务独立重置。

## 6. 导出到其他 MuJoCo 程序时

[`export_meta_policy_as_onnx()`](../humanoidverse/utils/helpers.py) 会从 checkpoint 的 actor input filter 推导 ONNX 输入顺序和维度；tracking 的 `--export-onnx true` 还会导出 backward encoder。23DoF 交付工具 [`export_wsl_bundle.py`](../humanoidverse/tools/export_wsl_bundle.py) 使用 `[state, last_action, history_actor, z]` 拼接 actor 输入，并记录观测、关节、动作缩放和 PD 控制契约。使用保存的 `z` 时，reward/goal 在任务切换前保持固定；tracking 按策略步读取 `zs_<motion_id>.npz` 中的 `z[step]`，从参考初始帧之后的第一个 transition 开始。
