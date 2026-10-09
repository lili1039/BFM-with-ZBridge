"""Reward inference for the MJLab backend.

This entrypoint avoids the legacy Isaac inference environment. Reward relabeling
uses the selected robot XML. The 23DoF release supports root and locomotion
reward tasks.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import numpy as np
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.goal_transition import TransitionConfig, optimize_transition
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.mjlab_reward_relabel import RewardWrapperHV
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_robot_config_manifest_args,
    add_bool_arg,
    checkpoint_load_device,
    configure_mediapy_ffmpeg,
    default_model_folder,
    load_mjlab_env_cfg,
    render_policy_frame,
    resolve_inference_data_and_robot_args,
    resolve_inference_robot_config,
    write_mjlab_relabel_xml,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


configure_mediapy_ffmpeg(media)


DEFAULT_TASKS = [
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
]

def _is_locomotion_task(task: str) -> bool:
    patterns = (
        r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$",
        r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$",
        r"^rotate-z-(-?\d+\.*\d*)-(\d+\.*\d*)$",
    )
    return any(re.search(pattern, task) for pattern in patterns)


def _resolve_reward_tasks(tasks: list[str] | None, robot_training) -> tuple[list[str], str]:
    selected_tasks = list(tasks or DEFAULT_TASKS)
    for task in selected_tasks:
        if not _is_locomotion_task(task):
            raise ValueError(
                f"Task {task} is not a supported locomotion reward for robot {robot_training.robot.name}."
            )
    return selected_tasks, "locomotion tasks"


def _export_model(model: torch.nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        z_dim=model.cfg.archi.z_dim,
    )
    print(f"[INFO] Exported model to {output_dir / f'{model_name}.onnx'}")


def _default_standing_target_states(wrapped_env, device: str) -> dict[str, torch.Tensor]:
    core_env = wrapped_env._env
    num_envs = int(core_env.num_envs)
    env_device = core_env.device

    init_state = core_env.config.robot.init_state
    root_pos = torch.as_tensor(init_state.pos, device=env_device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    if hasattr(core_env, "env_origins"):
        root_pos = root_pos + core_env.env_origins.to(device=env_device, dtype=torch.float32)
    root_rot_xyzw = torch.as_tensor(init_state.rot, device=env_device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    root_lin_vel = torch.zeros((num_envs, 3), device=env_device, dtype=torch.float32)
    root_ang_vel = torch.zeros((num_envs, 3), device=env_device, dtype=torch.float32)
    root_state_xyzw = torch.cat([root_pos, root_rot_xyzw, root_lin_vel, root_ang_vel], dim=-1)

    dof_state = torch.zeros((num_envs, core_env.num_dof, 2), device=env_device, dtype=torch.float32)
    dof_state[..., 0] = core_env.default_dof_pos.to(device=env_device, dtype=torch.float32)
    return {
        "root_states": root_state_xyzw.to(device=device, dtype=torch.float32),
        "dof_states": dof_state.to(device=device, dtype=torch.float32),
    }


def _load_replay_buffer(
    model_folder: Path,
    *,
    buffer_rank: int,
    buffer_path: Path | None,
) -> tuple[object, Path]:
    if buffer_path is not None:
        buffer_path = buffer_path.expanduser().resolve()
        if not buffer_path.is_dir():
            raise FileNotFoundError(f"Missing replay buffer path: {buffer_path}")
    else:
        buffers_dir = model_folder / "checkpoint" / "buffers"
        reduced = buffers_dir / "train_reduced"
        old_single_rank = buffers_dir / "train"
        rank_shard = buffers_dir / f"train_rank_{buffer_rank}"
        if reduced.is_dir():
            buffer_path = reduced
        elif rank_shard.is_dir():
            buffer_path = rank_shard
        elif old_single_rank.is_dir():
            buffer_path = old_single_rank
        else:
            raise FileNotFoundError(
                "Could not find replay buffer. Tried "
                f"{reduced}, {rank_shard}, and {old_single_rank}."
            )

    config_path = buffer_path / "config.json"
    if config_path.exists() and "TrajectoryDictBufferMultiDim" in config_path.read_text():
        dataset = TrajectoryDictBufferMultiDim.load(buffer_path, device="cpu")
        # Sampling a one-off CPU buffer does not benefit from compiling the
        # trajectory indexer; eager indexing avoids a lengthy Inductor build.
        dataset._get_idxs = get_idxs
    else:
        dataset = DictBuffer.load(buffer_path, device="cpu")
    return dataset, buffer_path


@torch.no_grad()
def _rollout_reward_sequences(
    *,
    model: torch.nn.Module,
    z_dict: dict[str, list[torch.Tensor]],
    tasks: list[str],
    wrapped_env: Any,
    env_cfg: Any,
    renderer: MujocoQposRenderer | None,
    output_dir: Path,
    device: str,
    episode_length: int,
    transition_mode: str,
    transition_config: TransitionConfig,
    save_mp4: bool,
    fps: int,
    latent_source: str | None = None,
) -> None:
    """Run each inferred reward vector in one continuous task sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    live = wrapped_env._env
    planner_env = None
    try:
        if transition_mode == "cem":
            planner_env, _ = env_cfg.build(num_envs=transition_config.candidates)
            planner_core = planner_env._env
            planner_core._motion_lib.load_motions_for_evaluation(start_idx=0)
            planner_core.is_evaluating = True
        for inference_idx in range(len(z_dict[tasks[0]])):
            task_z = [z_dict[task][inference_idx].detach().cpu().numpy().astype(np.float32).reshape(-1) for task in tasks]
            all_latents = np.stack(task_z)
            target_states = _default_standing_target_states(wrapped_env, device=device)
            observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
            initial_joint_pos = live.dof_pos[0].detach().cpu().numpy().copy()
            frames: list[np.ndarray] = []
            joint_pos: list[np.ndarray] = []
            joint_vel: list[np.ndarray] = []
            torques: list[np.ndarray] = []
            latents: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            task_indices: list[int] = []
            switches: list[dict[str, Any]] = []
            previous_z: torch.Tensor | None = None
            z_path: np.ndarray | None = None
            termination_reason = None
            use_env_render = True
            for step in range(episode_length * len(tasks)):
                task_idx, local_step = divmod(step, episode_length)
                if local_step == 0:
                    next_z = torch.as_tensor(task_z[task_idx], device=device).reshape(1, -1)
                    event: dict[str, Any] = {
                        "step": step,
                        "from": tasks[task_idx - 1] if task_idx else None,
                        "to": tasks[task_idx],
                        "start_joint_pos": live.dof_pos[0].detach().cpu().tolist(),
                    }
                    z_path = None
                    if task_idx and transition_mode == "cem":
                        assert previous_z is not None and planner_env is not None
                        z_path, plan_info = optimize_transition(
                            model,
                            wrapped_env,
                            observation,
                            planner_env,
                            previous_z[0].detach().cpu().numpy(),
                            task_z[task_idx],
                            all_latents,
                            None,
                            transition_config.steps,
                            transition_config,
                        )
                        event["planner"] = plan_info
                        if not plan_info["predicted"]["feasible"]:
                            termination_reason = "planner_no_feasible_path"
                            switches.append(event)
                            print(f"[INFO] No valid CEM path for {tasks[task_idx]}; stopping reward sequence.")
                            break
                    switches.append(event)
                    print(f"[INFO] Reward sequence switch step={step} task={tasks[task_idx]}")
                if z_path is not None and local_step < len(z_path):
                    z = torch.as_tensor(z_path[local_step:local_step + 1], device=device)
                else:
                    z = next_z
                action = model.act(observation, z, mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    termination_reason = "terminated" if bool(torch.as_tensor(terminated).any()) else "truncated"
                    print(f"[INFO] Reward sequence ended at step={step}: {termination_reason}")
                    break
                previous_z = z
                joint_pos.append(live.dof_pos[0].detach().cpu().numpy().copy())
                joint_vel.append(live.dof_vel[0].detach().cpu().numpy().copy())
                torques.append(live.torques[0].detach().cpu().numpy().copy())
                latents.append(z[0].detach().cpu().numpy().copy())
                actions.append(action[0].detach().cpu().numpy().copy())
                task_indices.append(task_idx)
                if save_mp4:
                    frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                    frames.append(frame)

            name = f"reward_sequence_{transition_mode}_{inference_idx}"
            trace_path = output_dir / f"{name}_trace.npz"
            q = np.asarray(joint_pos, dtype=np.float32)
            v = np.asarray(joint_vel, dtype=np.float32)
            tau = np.asarray(torques, dtype=np.float32)
            used_z = np.asarray(latents, dtype=np.float32)
            for event in switches[1:]:
                start = int(event["step"])
                end = min(start + transition_config.steps, len(q))
                if start >= end:
                    continue
                window_q = q[start:end]
                window_v = v[start:end]
                window_tau = tau[start:end]
                before_q = q[start - 1] if start else initial_joint_pos
                event["actual"] = {
                    "steps": end - start,
                    "first_step_z_jump_l2": float(np.linalg.norm(used_z[start] - used_z[start - 1])),
                    "rms_joint_speed_rad_s": float(np.sqrt(np.mean(window_v ** 2))),
                    "peak_joint_speed_rad_s": float(np.max(np.abs(window_v))),
                    "mean_joint_travel_rad": float(np.mean(np.sum(np.abs(np.diff(np.vstack([before_q, window_q]), axis=0)), axis=0))),
                    "peak_abs_joint_torque_nm": float(np.max(np.abs(window_tau))),
                }
            np.savez_compressed(
                trace_path,
                joint_pos=q,
                joint_vel=v,
                torque=tau,
                z=used_z,
                action=np.asarray(actions, dtype=np.float32),
                task_index=np.asarray(task_indices, dtype=np.int64),
                initial_joint_pos=initial_joint_pos,
            )
            video_path = None
            if save_mp4 and frames:
                video_path = output_dir / f"{name}.mp4"
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved reward sequence video: {video_path}")
            summary = {
                "mode": transition_mode,
                "reward_latent_source": latent_source or "replay_buffer_inference",
                "tasks": tasks,
                "inference_index": inference_idx,
                "steps": len(joint_pos),
                "dt_seconds": float(live.dt),
                "task_switch_interval": episode_length,
                "transition_steps": transition_config.steps if transition_mode == "cem" else 0,
                "cost_weights": {
                    "rms_joint_speed": transition_config.velocity_weight,
                    "peak_joint_speed": transition_config.peak_velocity_weight,
                    "joint_travel": transition_config.travel_weight,
                    "peak_joint_torque": transition_config.torque_weight,
                } if transition_mode == "cem" else None,
                "planner_constraint": "no_simulation_failure" if transition_mode == "cem" else None,
                "switches": switches,
                "termination_reason": termination_reason,
                "trace": str(trace_path),
                "video": str(video_path) if video_path else None,
            }
            (output_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        if planner_env is not None:
            planner_env.close()


def run_reward_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    episode_length: int,
    num_samples: int,
    n_inferences: int,
    skip_rollouts: bool,
    tasks: list[str],
    buffer_rank: int,
    buffer_path: Path | None,
    max_workers: int,
    process_executor: bool,
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_episode_length_s: float,
    export_onnx: bool,
    transition_mode: str = "independent",
    transition_steps: int = 40,
    cem_candidates: int = 8,
    cem_iterations: int = 3,
    cem_knots: int = 3,
    cem_basis_dim: int = 2,
    transition_seed: int = 0,
    transition_output_dir: Path | None = None,
    reward_latents_path: Path | None = None,
) -> None:
    if transition_mode not in {"independent", "hard", "cem"}:
        raise ValueError(f"Unknown transition mode: {transition_mode}")
    if episode_length < 1:
        raise ValueError("--episode-length must be positive")
    if transition_mode == "cem" and not skip_rollouts and (not disable_dr or not disable_obs_noise):
        raise ValueError("CEM planning requires --disable-dr true and --disable-obs-noise true")
    transition_config = TransitionConfig(
        steps=transition_steps,
        candidates=cem_candidates,
        iterations=cem_iterations,
        knots=cem_knots,
        basis_dim=cem_basis_dim,
        seed=transition_seed,
    )
    if transition_mode == "cem" and not skip_rollouts:
        transition_config.validate(episode_length)
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = resolve_inference_robot_config(robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    tasks, task_support_mode = _resolve_reward_tasks(tasks, robot_training)
    if transition_mode in {"hard", "cem"} and not skip_rollouts and len(tasks) < 2:
        raise ValueError("Continuous reward transitions require at least two tasks")

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    if export_onnx:
        _export_model(model, model_folder / "exported")

    output_dir = model_folder / "reward_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    latent_source = None
    if reward_latents_path is None:
        print("[INFO] Loading replay buffer...", end=" ", flush=True)
        start_t = time.time()
        dataset, loaded_buffer_path = _load_replay_buffer(model_folder, buffer_rank=buffer_rank, buffer_path=buffer_path)
        print(f"done in {time.time() - start_t:.2f}s")
        print(f"[INFO] Replay buffer={loaded_buffer_path}")
        if hasattr(dataset, "size"):
            print(f"[INFO] Replay buffer sampled transition count={dataset.size()}")
        relabel_xml = write_mjlab_relabel_xml(
            robot_xml, output_dir, control_joint_names, robot_training.robot.name,
            root_body_name=robot_training.robot.base_body,
        )
        reward_eval_agent = RewardWrapperHV(
            model=model, inference_dataset=dataset,
            num_samples_per_inference=int(num_samples), inference_function="reward_wr_inference",
            max_workers=int(max_workers), process_executor=bool(process_executor), env_model=str(relabel_xml),
        )
    else:
        latent_source = str(reward_latents_path.expanduser().resolve())
        if not Path(latent_source).is_file():
            raise FileNotFoundError(latent_source)
        print(f"[INFO] Reusing reward embeddings from {latent_source}")

    print(f"[INFO] UFO reward inference model_folder={model_folder}")
    print(f"[INFO] Robot={robot_training.robot.name}")
    print(f"[INFO] Robot config={Path(robot_training.config_path).expanduser().resolve()}")
    print(f"[INFO] num_dof={num_dof}")
    print(f"[INFO] Reward source XML={robot_xml}")
    if reward_latents_path is None:
        print(f"[INFO] Reward relabel XML={relabel_xml}")
    print(f"[INFO] task support mode={task_support_mode}")
    print(f"[INFO] device={device} save_mp4={save_mp4} skip_rollouts={skip_rollouts}")
    print(f"[INFO] tasks={tasks}")

    z_dict: dict[str, list[torch.Tensor]] = {}
    output_path = output_dir / "reward_locomotion.pkl"
    if reward_latents_path is None:
        for inference_idx in range(int(n_inferences)):
            for task in tasks:
                print(f"[INFO] Started reward inference {inference_idx + 1}/{n_inferences} for {task}...", end=" ", flush=True)
                start_t = time.time()
                z = reward_eval_agent.reward_inference(task=task)
                z_dict.setdefault(task, []).append(z.detach().cpu())
                print(f"done in {time.time() - start_t:.2f}s")
                joblib.dump(z_dict, output_path)
                print(f"[INFO] Saved reward embeddings: {output_path}")
    else:
        stored = joblib.load(latent_source)
        for task in tasks:
            if task not in stored or len(stored[task]) < int(n_inferences):
                raise ValueError(f"Cached reward embeddings do not contain {n_inferences} vector(s) for {task}")
            z_dict[task] = [torch.as_tensor(z, dtype=torch.float32).reshape(1, -1).cpu()
                            for z in stored[task][:int(n_inferences)]]
            if any(z.shape[-1] != model.cfg.archi.z_dim or not torch.isfinite(z).all() for z in z_dict[task]):
                raise ValueError(f"Cached reward embeddings have invalid shape or values for {task}")
        if output_path.resolve() != Path(latent_source):
            joblib.dump(z_dict, output_path)

    if not z_dict:
        raise RuntimeError("No reward embeddings were generated")
    reward_latents = np.stack(
        [np.concatenate([z.detach().cpu().numpy().astype(np.float32) for z in z_dict[task]], axis=0) for task in tasks],
        axis=0,
    )
    if reward_latents.shape[1] == 1:
        reward_latents = reward_latents[:, 0, :]
    np.savez_compressed(output_dir / "reward_latents.npz", names=np.asarray(tasks, dtype=np.str_), latents=reward_latents)

    if skip_rollouts:
        return

    env_cfg, _use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    renderer = None
    try:
        print(f"[INFO] Generating rollout videos with XML={env_cfg.mjcf_path}")
        if save_mp4:
            renderer = MujocoQposRenderer(
                robot_xml,
                render_size=render_size,
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                expected_qpos_size=7 + num_dof,
            )
        if transition_mode != "independent":
            _rollout_reward_sequences(
                model=model,
                z_dict=z_dict,
                tasks=tasks,
                wrapped_env=wrapped_env,
                env_cfg=env_cfg,
                renderer=renderer,
                output_dir=transition_output_dir.expanduser().resolve() if transition_output_dir is not None else output_dir / "transitions",
                device=device,
                episode_length=episode_length,
                transition_mode=transition_mode,
                transition_config=transition_config,
                save_mp4=save_mp4,
                fps=fps,
                latent_source=latent_source,
            )
            return
        for task in tasks:
            frames = []
            for z_cpu in z_dict[task]:
                z = z_cpu.to(device).repeat(1, 1)
                target_states = _default_standing_target_states(wrapped_env, device=device)
                observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
                print("[INFO] Reset reward rollout to default standing pose.")
                use_env_render = True
                if save_mp4:
                    frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                    frames.append(frame)
                for step in range(int(episode_length)):
                    action = model.act(observation, z, mean=True)
                    observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                    if save_mp4:
                        frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                        frames.append(frame)
                    if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                        print(f"[INFO] Task {task} episode ended at step={step}; stopping this rollout.")
                        break
            if save_mp4:
                video_path = video_dir / f"{task}.mp4"
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved reward rollout video for {task}: {video_path}")
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO reward inference.")
    parser.add_argument("--model-folder", type=Path, default=default_model_folder(),
                        help="Trained run directory; defaults to the stable 23DoF demo checkpoint.")
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="reward inference")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save policy rollout MP4s.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=150_000)
    parser.add_argument("--n-inferences", type=int, default=1)
    add_bool_arg(parser, "--skip-rollouts", False, "Only compute reward embeddings; do not create rollout videos.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task subset. Defaults to the full locomotion task list.")
    parser.add_argument("--transition-mode", choices=("independent", "hard", "cem"), default="independent",
                        help="Independent task resets, abrupt continuous switches, or CEM-planned continuous switches.")
    parser.add_argument("--transition-steps", type=int, default=40,
                        help="CEM z-path duration in policy steps; --episode-length is the task-switch interval.")
    parser.add_argument("--cem-candidates", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-knots", type=int, default=3)
    parser.add_argument("--cem-basis-dim", type=int, default=2)
    parser.add_argument("--transition-seed", type=int, default=0)
    parser.add_argument("--transition-output-dir", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0, help="Rank-local replay buffer shard to use, e.g. train_rank_0.")
    parser.add_argument("--buffer-path", type=Path, default=None, help="Explicit replay buffer directory; overrides --buffer-rank.")
    parser.add_argument("--reward-latents-path", type=Path, default=None,
                        help="Reuse reward embeddings from a matching checkpoint instead of relabeling the replay buffer.")
    parser.add_argument("--max-workers", type=int, default=24)
    add_bool_arg(parser, "--process-executor", True, "Use ProcessPoolExecutor for reward relabel workers.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", False, "Export ONNX next to the checkpoint before inference.")
    return resolve_inference_data_and_robot_args(parser.parse_args(), parser)


def main() -> None:
    args = parse_args()
    run_reward_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        episode_length=args.episode_length,
        num_samples=args.num_samples,
        n_inferences=args.n_inferences,
        skip_rollouts=args.skip_rollouts,
        tasks=args.tasks,
        buffer_rank=args.buffer_rank,
        buffer_path=args.buffer_path,
        max_workers=args.max_workers,
        process_executor=args.process_executor,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
        transition_mode=args.transition_mode,
        transition_steps=args.transition_steps,
        cem_candidates=args.cem_candidates,
        cem_iterations=args.cem_iterations,
        cem_knots=args.cem_knots,
        cem_basis_dim=args.cem_basis_dim,
        transition_seed=args.transition_seed,
        transition_output_dir=args.transition_output_dir,
        reward_latents_path=args.reward_latents_path,
    )


if __name__ == "__main__":
    main()
