# UFO-FB G1 23DoF：训练、可视化与 WSL/MuJoCo 交付教程

本文介绍 Unitree G1 23DoF 上的 FB 训练、MuJoCo 可视化和 WSL 导出。除 WSL 解压与校验外，命令均从仓库根目录执行。大型数据和 checkpoint 放在仓库外；下文约定其目录为 `../data/UFO/`，通过 `UFO_DATA_ROOT` 引用。数据位于其他位置时，在执行命令前设置该变量。

推理、导出和仓库演示使用 `ufo_fb_g1_23dof_4096_seed4728_stable` checkpoint（102,498,304 transitions / 1,600,640 optimizer steps）。

## 1. 环境与数据

在仓库根目录设置外置数据目录并安装依赖：

```bash
export UFO_DATA_ROOT="${UFO_DATA_ROOT:-$(pwd)/../data/UFO}"
export UFO_CACHE_DIR="$UFO_DATA_ROOT/cache"
export UFO_MODEL="$UFO_DATA_ROOT/runs/ufo_fb_g1_23dof_4096_seed4728_stable"
export CUDA_VISIBLE_DEVICES=0
UV_CACHE_DIR="$UFO_CACHE_DIR/uv" UV_LINK_MODE=copy uv sync --frozen
```

查看显卡状态：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

`CUDA_VISIBLE_DEVICES=0` 选择物理 GPU 0；进程内部使用 `cuda:0`。

`humanoidverse/tools/retarget_g1_29dof_to_23dof.py` 将原始 LaFAN 动作转换为 40 条、30 Hz 的 23DoF NPZ，保存在 `$UFO_DATA_ROOT/lafan_g1_23dof_ik/npz/`；质量报告位于同级 `reports/retarget_report.json`。数据配置 `configs/data/lafan_g1_23dof_ik.yaml` 生成 40 条完整动作和 902 条约 10 秒的训练切片。23DoF XML 与 mesh 位于 `humanoidverse/data/robots/g1_23dof/`。重建 retarget 数据时，给脚本传入 `--source-xml`、`--input-dir` 和 `--output-dir`；重建 motion cache 时给训练命令添加 `--rebuild-motion-cache`。

23DoF 定义是腿 12、腰 yaw 1、左右手臂各 5 个关节；训练与推理都使用匹配的 23DoF 数据和 checkpoint。机器人顺序和 PD 参数以 `configs/robots/g1_23dof.yaml` 为准。

## 2. 训练与续训

`run_train.sh` 配置运行环境后调用 `humanoidverse.train`。`--num-env-steps` 表示从训练开始累计的 transitions 上限；`--work-dir` 指定训练输出目录，目录内有 checkpoint 时会自动续训。训练入口的默认输出目录是 `runs/ufo_fb_g1_23dof_new`。三个推理入口默认读取 `$UFO_DATA_ROOT/runs/ufo_fb_g1_23dof_4096_seed4728_stable`，也可通过 `UFO_MODEL` 或 `--model-folder` 指定模型。

| 每卡并行环境 | FB 每次触发的 optimizer updates | warm-up transitions | replay 时间槽/环境 |
|---:|---:|---:|---:|
| 1,024 | 16 | 10,240 | 5,000 |
| 2,048 | 32 | 20,480 | 2,500 |
| 4,096 | 64 | 40,960 | 1,250 |

表中的配置维持近似相同的更新密度。以下示例使用 4,096 个环境、192M transitions、每卡 5,120,000 条 replay capacity，每约 3.2M transitions 评估和保存一次。调整环境数时，按表同步调整 `--num-agent-updates`。

新建 4,096 环境的 FB 训练：

```bash
export UFO_TRAIN_RUN="$UFO_DATA_ROOT/runs/ufo_fb_g1_23dof_4096_seed4730"
mkdir -p "$UFO_TRAIN_RUN"
PYTHONUNBUFFERED=1 WANDB_MODE=online ./run_train.sh --agent fb --gpu-ids single \
  --data-manifest configs/data/lafan_g1_23dof_ik.yaml --robot-config configs/robots/g1_23dof.yaml \
  --num-envs 4096 --num-env-steps 192000000 --num-agent-updates 64 \
  --update-z-every-step 100 --buffer-size 5120000 --checkpoint-every-steps 3200000 \
  --eval-video-motion-ids 7 165 390 465 620 --eval-video-max-steps 300 \
  --eval-video-render-size 256 --seed 4730 --clip-grad-norm 1.0 \
  --use-wandb --wandb-run-name ufo_fb_g1_23dof_4096_seed4730 \
  --work-dir "$UFO_TRAIN_RUN" > "$UFO_TRAIN_RUN/console.log" 2>&1
```

`--use-wandb` 与 `WANDB_MODE=online` 上传指标和评估视频；`--smoke` 运行最多 16 个环境的快速检查。

从 stable checkpoint 续训到累计 192M 时，先复制 checkpoint、optimizer 和 replay 到新目录：

```bash
export UFO_CONTINUATION_RUN="$UFO_DATA_ROOT/runs/ufo_fb_g1_23dof_from102m_to192m"
uv run python -m humanoidverse.tools.stage_continuation --source "$UFO_MODEL" --work-dir "$UFO_CONTINUATION_RUN"
```

然后在新目录启动训练。`192000000` 是累计 transitions 终点。

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=online ./run_train.sh --agent fb --gpu-ids single \
  --data-manifest configs/data/lafan_g1_23dof_ik.yaml --robot-config configs/robots/g1_23dof.yaml \
  --num-envs 4096 --num-env-steps 192000000 --num-agent-updates 64 \
  --update-z-every-step 100 --buffer-size 5120000 --checkpoint-every-steps 3200000 \
  --eval-video-motion-ids 7 165 390 465 620 --eval-video-max-steps 300 \
  --eval-video-render-size 256 --seed 4728 --clip-grad-norm 1.0 \
  --use-wandb --wandb-run-name ufo_fb_g1_23dof_from102m_to192m \
  --work-dir "$UFO_CONTINUATION_RUN" > "$UFO_CONTINUATION_RUN/console.log" 2>&1
```

## 3. 看进度、W&B 与评估视频

依次查看训练进程、已保存的 checkpoint 进度和最近的控制台日志：

```bash
pgrep -af '[h]umanoidverse.train'
```

```bash
sed -n '1,80p' "$UFO_MODEL/checkpoint/train_status.json"
```

```bash
tail -n 40 "$UFO_MODEL/console.log"
```

`train_log.txt` 记录训练指标，`humanoidverse_tracking_eval.csv` 记录逐动作评估。W&B 中的训练指标位于 `train/*`，评估指标位于 `eval/humanoidverse_tracking_eval/*`。

训练评估视频使用训练切片 ID `7 165 390 465 620`，分别展示 dance、get-up、jump、run 和 walk。每段最多 300 帧、50 FPS；左侧是参考动作，右侧是策略动作。本地视频位于 `<work-dir>/videos/eval_<step>/`，W&B 视频位于 `eval/humanoidverse_tracking_eval/video/*`。

评估时查看 `distance`、`emd`（越低越好）、`proximity`（越高越好）及视频中的动作幅度、平衡和脚滑。`mpjpe_l` 在当前实现中是关节角误差，单位为 mrad；`B_norm`、`z_norm` 通常约为 16。

## 4. 在服务器上生成 MuJoCo 可视化

以下命令使用 stable checkpoint。Dance tracking 使用训练切片 ID 7（`dance1_subject1__clip007`）；goal/reward 使用 40 条完整动作的数据集，motion ID 按该数据集编号。命令中的 `--disable-dr true` 和 `--disable-obs-noise true` 关闭随机扰动，使重复运行结果可比。

Tracking：播放参考舞蹈并运行策略，生成参考/策略并排 MP4，输出到 `<model-folder>/tracking_inference/tracking_7.mp4`：

```bash
uv run python -m humanoidverse.tracking_inference \
  --model-folder "$UFO_MODEL" \
  --data-path "$UFO_DATA_ROOT/cache/motion_data/lafan_g1_23dof_ik/lafan_g1_23dof_ik_train_near10s_ufo.pkl" \
  --robot-config configs/robots/g1_23dof.yaml --device cuda:0 \
  --headless true --save-mp4 true --disable-dr true --disable-obs-noise true \
  --motion-list 7 --max-steps 300 --render-size 256 --export-onnx false
```

Goal reaching：目标帧定义在 `configs/goals/lafan_g1_23dof_ik.json`。下例按 7 个目标运行 280 步，每 40 步切换目标；CEM 用 30 步完成一次 `z` 切换：

```bash
uv run python -m humanoidverse.goal_inference \
  --model-folder "$UFO_MODEL" --data-manifest configs/data/lafan_g1_23dof_ik.yaml \
  --dataset lafan_g1_23dof_ik --robot-config configs/robots/g1_23dof.yaml \
  --goal-json configs/goals/lafan_g1_23dof_ik.json --goal-indices 13 11 12 11 13 4 5 \
  --device cuda:0 --headless true --save-mp4 true --disable-dr true --disable-obs-noise true \
  --episode-len 280 --goal-switch-interval 40 --transition-mode cem --transition-steps 30 \
  --cem-candidates 6 --cem-iterations 2 --cem-knots 3 --cem-basis-dim 2 \
  --goal-tolerance 0.25 --render-size 224 \
  --transition-output-dir "$UFO_DATA_ROOT/goal_transition_experiments/repro_7_goals" --export-onnx false
```

Reward inference：读取 stable checkpoint 生成的任务 `z`，依次运行前进、侧移、旋转、后退、站立和低姿态站立。需要重新计算任务 `z` 时，去掉 `--reward-latents-path`；程序会从该 checkpoint 的 replay 中采样，默认 150,000 条：

```bash
uv run python -m humanoidverse.reward_inference \
  --model-folder "$UFO_MODEL" --data-manifest configs/data/lafan_g1_23dof_ik.yaml \
  --dataset lafan_g1_23dof_ik --robot-config configs/robots/g1_23dof.yaml \
  --reward-latents-path "$UFO_MODEL/reward_inference/reward_locomotion.pkl" \
  --tasks move-ego-0-0.3 move-ego-90-0.3 rotate-z-5-0.5 \
  move-ego-180-0.3 move-ego-0-0 move-ego-low0.5-0-0 \
  --device cuda:0 --headless true --save-mp4 true --disable-dr true --disable-obs-noise true \
  --episode-length 80 --transition-mode cem --transition-steps 30 \
  --cem-candidates 6 --cem-iterations 2 --cem-knots 3 --cem-basis-dim 2 \
  --render-size 224 --transition-output-dir "$UFO_DATA_ROOT/reward_transition_experiments/repro_6_tasks" \
  --export-onnx false
```

制作对照视频时，用相同参数和输出目录再运行一次 `--transition-mode hard`，然后运行 `humanoidverse.tools.compare_latent_transitions`；合成命令见[根目录 README](../README.md)。输出包括并排视频、`z` 球面与关节力矩动画、指标图、CSV 和 JSON。查看已有的 [goal 对照](../assets/demos/goal/comparison.mp4)、[reward 对照](../assets/demos/reward/comparison.mp4)、[goal 球面/力矩](../assets/demos/goal/z_sphere.mp4)及 [reward 球面/力矩](../assets/demos/reward/z_sphere.mp4)。

## 5. 导出并下载 WSL sim-to-sim 验证包

`humanoidverse.tools.export_wsl_bundle` 从指定 checkpoint 生成 9 个 reward、14 个 goal 的静态 256 维 `z`，以及完整动作 ID `2 9 17 22 28` 的逐步 tracking `z`。交付包还包括 ONNX、模型 safetensors、机器人 XML/mesh、MuJoCo 控制契约、README、manifest、校验和及 `.tar.gz`。生成 reward `z` 时会读取源 checkpoint 的 replay buffer。

预览导出路径与命令：

```bash
uv run python -m humanoidverse.tools.export_wsl_bundle --checkpoint "$UFO_MODEL" --dry-run
```

执行导出：

```bash
uv run python -m humanoidverse.tools.export_wsl_bundle --checkpoint "$UFO_MODEL"
```

导出目录为 `$UFO_DATA_ROOT/exports/ufo_fb_g1_23dof_4096_seed4728_stable_step102498304_wsl`，旁边生成同名 `.tar.gz`。通过 `--output-dir` 可指定其他目录；`--no-safetensors` 可省略模型权重副本，`--no-archive` 可省略压缩包。将压缩包下载到 WSL 后，在下载目录解压：

```bash
tar -xzf ufo_fb_g1_23dof_4096_seed4728_stable_step102498304_wsl.tar.gz
```

```bash
cd ufo_fb_g1_23dof_4096_seed4728_stable_step102498304_wsl && sha256sum -c SHA256SUMS --quiet
```

校验完成后，按包内 `README.md` 操作。`model/control_contract.json` 定义 MuJoCo 控制接口，`manifest.json` 列出文件、动作和哈希。

在 WSL 的 MuJoCo runner 中按 `model/control_contract.json` 组装策略输入：`[state(52), last_action(23), history_actor(300), z(256)]`，共 631 维。History 按字段分组，各字段最新帧在前；物理频率 200 Hz，策略频率 50 Hz；根四元数采用 `wxyz`。将策略动作按控制契约完成缩放、裁剪、目标角计算和 PD/力矩限幅，再写入仿真控制。Reward/goal 使用固定任务 `z`，tracking 按 50 Hz 更新 `z`。随后对照视频和 `distance/emd` 指标评估运行效果。
