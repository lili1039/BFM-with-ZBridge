"""Goal inference and video export for UFO policies.

This entrypoint avoids the legacy Isaac inference path.  Goal embeddings are
computed from MJLab motion observations, and optional videos are rendered from
MJLab rollout state with pure MuJoCo qpos fallback.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.goal_transition import TransitionConfig, optimize_transition
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation
from humanoidverse.mjlab_inference_utils import (
    PROJECT_ROOT,
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
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


configure_mediapy_ffmpeg(media)


def _find_goal_json(goal_json: Path | None, *, num_dof: int, robot_name: str) -> Path:
    if goal_json is not None:
        goal_json = goal_json.expanduser().resolve()
        if not goal_json.exists():
            raise FileNotFoundError(f"Missing goal JSON: {goal_json}")
        return goal_json

    if int(num_dof) == 23 and robot_name == "g1_23dof":
        default_goal_json = PROJECT_ROOT / "configs/goals/lafan_g1_23dof_ik.json"
        if default_goal_json.exists():
            return default_goal_json
        raise FileNotFoundError(f"Missing default 23DoF goal JSON: {default_goal_json}")
    raise ValueError(f"Goal inference for {robot_name} requires --goal-json matching the selected robot.")


_GOAL_DOF_KEYS = {
    "dof_pos",
    "joint_pos",
    "joint_positions",
    "target_dof_pos",
    "target_joint_pos",
    "target_joint_positions",
}
_GOAL_QPOS_KEYS = {"qpos", "target_qpos"}


def _numeric_last_dim(value: object) -> int | None:
    try:
        array = np.asarray(value)
    except ValueError:
        return None
    if array.ndim == 0 or array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return None
    return int(array.shape[-1])


def _validate_goal_value_dim(value: object, *, expected_dim: int, key_path: str, goal_json: Path) -> None:
    dim = _numeric_last_dim(value)
    if dim is not None and dim != int(expected_dim):
        raise ValueError(f"Goal JSON {goal_json} field {key_path} expected dimension {expected_dim}, got {dim}")


def _validate_goal_entry_dims(value: object, *, num_dof: int, goal_json: Path, key_path: str = "goal") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{key_path}.{key}"
            if key in _GOAL_DOF_KEYS:
                _validate_goal_value_dim(child, expected_dim=num_dof, key_path=child_path, goal_json=goal_json)
            elif key in _GOAL_QPOS_KEYS:
                _validate_goal_value_dim(child, expected_dim=7 + int(num_dof), key_path=child_path, goal_json=goal_json)
            else:
                _validate_goal_entry_dims(child, num_dof=num_dof, goal_json=goal_json, key_path=child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            if isinstance(child, (dict, list)):
                _validate_goal_entry_dims(child, num_dof=num_dof, goal_json=goal_json, key_path=f"{key_path}[{idx}]")


def load_and_validate_goal_json(goal_json: Path, *, num_dof: int) -> list[dict[str, object]]:
    with Path(goal_json).open("r") as f:
        goals_to_evaluate = json.load(f)
    if not goals_to_evaluate:
        raise RuntimeError("Goal JSON is empty.")
    if not isinstance(goals_to_evaluate, list):
        raise ValueError(f"Goal JSON must be a list of goal entries: {goal_json}")
    for idx, goal in enumerate(goals_to_evaluate):
        if not isinstance(goal, dict):
            raise ValueError(f"Goal JSON entry #{idx} must be a mapping: {goal_json}")
        _validate_goal_entry_dims(goal, num_dof=int(num_dof), goal_json=Path(goal_json), key_path=f"goal[{idx}]")
    return goals_to_evaluate


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


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str, *, num_dof: int) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}


def run_goal_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    goal_json: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    episode_len: int,
    goal_switch_interval: int,
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_episode_length_s: float,
    export_onnx: bool,
    goal_indices: list[int] | None = None,
    transition_mode: str = "hard",
    transition_steps: int = 40,
    cem_candidates: int = 8,
    cem_iterations: int = 3,
    cem_knots: int = 3,
    cem_basis_dim: int = 2,
    goal_tolerance: float = 0.25,
    transition_seed: int = 0,
    transition_output_dir: Path | None = None,
) -> None:
    if transition_mode not in {"hard", "cem"}:
        raise ValueError(f"Unknown transition mode: {transition_mode}")
    if transition_mode == "cem" and (not disable_dr or not disable_obs_noise):
        raise ValueError("CEM planning requires --disable-dr true and --disable-obs-noise true so candidate rollouts match the live state")
    transition_config = TransitionConfig(
        steps=transition_steps,
        candidates=cem_candidates,
        iterations=cem_iterations,
        knots=cem_knots,
        basis_dim=cem_basis_dim,
        goal_tolerance=goal_tolerance,
        seed=transition_seed,
    )
    if transition_mode == "cem":
        transition_config.validate(goal_switch_interval)
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
    goal_json_path = _find_goal_json(goal_json, num_dof=num_dof, robot_name=robot_training.robot.name)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
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
    env = wrapped_env._env

    output_dir = model_folder / "goal_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO goal inference model_folder={model_folder}")
    print(f"[INFO] Robot={robot_training.robot.name}")
    print(f"[INFO] Robot config={Path(robot_training.config_path).expanduser().resolve()}")
    print(f"[INFO] Robot XML={robot_xml}")
    print(f"[INFO] num_dof={num_dof}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Goal JSON={goal_json_path}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    try:
        if export_onnx:
            _export_model(model, model_folder / "exported")

        goals_to_evaluate = load_and_validate_goal_json(goal_json_path, num_dof=num_dof)
        z_dict: dict[str, object] = {}
        target_dof_dict: dict[str, torch.Tensor] = {}
        goal_source_dict: dict[str, dict[str, object]] = {}
        with torch.no_grad():
            pbar = tqdm(goals_to_evaluate, leave=False, disable=False)
            for goal in pbar:
                motion_id = int(goal["motion_id"])
                env.set_is_evaluating(motion_id)
                # A one-env evaluation loads the selected global motion into
                # local MotionLib slot 0.
                gobs, gobs_dict = get_backward_observation(
                    env,
                    0,
                    use_root_height_obs=use_root_height_obs,
                    velocity_multiplier=0,
                )
                num_frames = next(iter(gobs.values())).shape[0]
                frame_pbar = tqdm(goal["frames"], leave=False, disable=False, desc="frames")
                for frame_idx in frame_pbar:
                    frame_idx = int(frame_idx)
                    if frame_idx >= num_frames:
                        pbar.write(f"  Skipping frame_idx {frame_idx} (motion has {num_frames} frames)")
                        continue
                    goal_name = f"{goal['motion_name']}_{frame_idx}"
                    goal_observation = {key: value[frame_idx][None, ...] for key, value in gobs.items()}
                    goal_observation = tree_map(
                        lambda x: torch.as_tensor(x, device=device, dtype=torch.float32),
                        goal_observation,
                    )
                    z_dict[goal_name] = model.goal_inference(goal_observation).detach().cpu().numpy()
                    target_dof_dict[goal_name] = gobs_dict["dof_pos"][frame_idx].detach().clone()
                    goal_source_dict[goal_name] = goal

        output_path = output_dir / "goal_reaching.pkl"
        joblib.dump(z_dict, output_path)
        print(f"[INFO] Saved goal embeddings: {output_path} ({len(z_dict)} goals)")
        if not z_dict:
            raise RuntimeError("No goal embeddings were generated.")
        np.savez_compressed(
            output_dir / "goal_latents.npz",
            names=np.asarray(list(z_dict), dtype=np.str_),
            latents=np.concatenate([np.asarray(z_dict[name], dtype=np.float32) for name in z_dict], axis=0),
        )

        if not save_mp4 and transition_mode == "hard":
            return

        all_goal_names = list(z_dict)
        if goal_indices is not None:
            if not goal_indices or any(i < 0 or i >= len(all_goal_names) for i in goal_indices):
                raise ValueError(f"--goal-indices must be within [0, {len(all_goal_names) - 1}]")
            goal_names = [all_goal_names[i] for i in goal_indices]
        else:
            goal_names = all_goal_names
        if goal_switch_interval < 1:
            raise ValueError("--goal-switch-interval must be positive")
        if transition_mode == "cem" and len(goal_names) < 2:
            raise ValueError("CEM transition requires at least two selected goals")
        result_dir = (
            transition_output_dir.expanduser().resolve()
            if transition_output_dir is not None
            else video_dir
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        renderer = (
            MujocoQposRenderer(
                robot_xml,
                render_size=render_size,
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                expected_qpos_size=7 + num_dof,
            )
            if save_mp4 else None
        )
        planner_env = None
        try:
            first_goal_name = goal_names[0]
            first_goal = goal_source_dict[first_goal_name]
            first_motion_id = int(first_goal["motion_id"])
            env.set_is_evaluating(first_motion_id)
            # The selected global motion is exposed through local slot 0; see
            # the embedding loop above.
            _first_backward_obs, first_obs_dict = get_backward_observation(
                env,
                0,
                use_root_height_obs=use_root_height_obs,
                velocity_multiplier=0,
            )
            target_states = _target_states_from_obs(first_obs_dict, device=device, num_dof=num_dof)
            observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
            first_motion_name = first_goal.get("motion_name", "unknown")
            print(
                f"[INFO] Reset goal rollout to demo start: "
                f"motion_id={first_motion_id}, motion_name={first_motion_name}, frame=0"
            )
            frames = []
            goal_idx = -1
            z = None
            previous_z = None
            z_path = None
            switch_events: list[dict[str, object]] = []
            joint_pos_trace: list[np.ndarray] = []
            joint_vel_trace: list[np.ndarray] = []
            torque_trace: list[np.ndarray] = []
            latent_trace: list[np.ndarray] = []
            action_trace: list[np.ndarray] = []
            goal_error_trace: list[float] = []
            termination_reason = None
            use_env_render = True
            if transition_mode == "cem":
                planner_env, _ = env_cfg.build(num_envs=cem_candidates)
                planner_core = planner_env._env
                planner_core._motion_lib.load_motions_for_evaluation(start_idx=first_motion_id)
                planner_core.is_evaluating = True
                all_goal_z = np.concatenate([np.asarray(z_dict[name], dtype=np.float32) for name in all_goal_names], axis=0)
            for step in tqdm(range(int(episode_len)), desc="steps", leave=False):
                if step % int(goal_switch_interval) == 0:
                    goal_idx = (goal_idx + 1) % len(goal_names)
                    print(f"[INFO] Switching to goal {goal_names[goal_idx]} at step {step}")
                    z_next = torch.as_tensor(z_dict[goal_names[goal_idx]], device=device, dtype=torch.float32)
                    event: dict[str, object] = {
                        "step": step,
                        "from": goal_names[goal_idx - 1] if step > 0 else None,
                        "to": goal_names[goal_idx],
                    }
                    if step > 0 and transition_mode == "cem":
                        assert previous_z is not None and planner_env is not None
                        z_path, plan_info = optimize_transition(
                            model,
                            wrapped_env,
                            observation,
                            planner_env,
                            previous_z.detach().cpu().numpy().reshape(-1),
                            z_next.detach().cpu().numpy().reshape(-1),
                            all_goal_z,
                            target_dof_dict[goal_names[goal_idx]].to(device),
                            goal_switch_interval,
                            transition_config,
                        )
                        event["planner"] = plan_info
                        print(f"[INFO] CEM transition cost={plan_info['cost']:.4f} predicted={plan_info['predicted']}")
                    else:
                        z_path = None
                    previous_z = z_next
                    event["start_joint_pos"] = env.dof_pos[0].detach().cpu().tolist()
                    switch_events.append(event)

                local_step = step % int(goal_switch_interval)
                if z_path is not None and local_step < len(z_path):
                    z = torch.as_tensor(z_path[local_step:local_step + 1], device=device)
                else:
                    z = previous_z

                action = model.act(observation, z.repeat(1, 1), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    termination_reason = "terminated" if bool(torch.as_tensor(terminated).any()) else "truncated"
                    print(f"[INFO] Episode ended at step={step}; stopping goal rollout video.")
                    break
                joint_pos_trace.append(env.dof_pos[0].detach().cpu().numpy().copy())
                joint_vel_trace.append(env.dof_vel[0].detach().cpu().numpy().copy())
                torque_trace.append(env.torques[0].detach().cpu().numpy().copy())
                latent_trace.append(z[0].detach().cpu().numpy().copy())
                action_trace.append(action[0].detach().cpu().numpy().copy())
                goal_error_trace.append(float((env.dof_pos[0] - target_dof_dict[goal_names[goal_idx]].to(device)).abs().mean().item()))
                if save_mp4:
                    frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                    frames.append(frame)

            video_path = None
            if save_mp4:
                video_path = result_dir / ("goal.mp4" if transition_output_dir is None and goal_indices is None and transition_mode == "hard" else f"goal_{transition_mode}.mp4")
                if not frames:
                    raise RuntimeError("Goal rollout ended before any valid frame was rendered")
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved goal video: {video_path}")
            q = np.asarray(joint_pos_trace, dtype=np.float32)
            v = np.asarray(joint_vel_trace, dtype=np.float32)
            tau = np.asarray(torque_trace, dtype=np.float32)
            zs = np.asarray(latent_trace, dtype=np.float32)
            actions = np.asarray(action_trace, dtype=np.float32)
            errors = np.asarray(goal_error_trace, dtype=np.float32)
            for event_idx, event in enumerate(switch_events):
                start = int(event["step"])
                end = int(switch_events[event_idx + 1]["step"]) if event_idx + 1 < len(switch_events) else len(q)
                if end <= start:
                    event["executed_steps"] = 0
                    event["terminated_early"] = termination_reason is not None
                    event.pop("start_joint_pos")
                    continue
                q_prev = np.asarray(event["start_joint_pos"], dtype=np.float32)
                interval_q = q[start:end]
                event["executed_steps"] = end - start
                event["terminated_early"] = bool(termination_reason is not None and end == len(q))
                event["final_joint_mae_rad"] = None if event["terminated_early"] else float(errors[end - 1])
                event["last_valid_joint_mae_rad"] = float(errors[end - 1])
                event["rms_joint_speed_rad_s"] = float(np.sqrt(np.mean(v[start:end] ** 2)))
                event["peak_joint_speed_rad_s"] = float(np.max(np.abs(v[start:end])))
                event["mean_joint_travel_rad"] = float(np.mean(np.sum(np.abs(np.diff(np.vstack([q_prev, interval_q]), axis=0)), axis=0)))
                event["mean_net_joint_displacement_rad"] = float(np.mean(np.abs(interval_q[-1] - q_prev)))
                event["peak_abs_joint_torque_nm"] = float(np.max(np.abs(tau[start:end])))
                event["first_step_z_change_l2"] = float(np.linalg.norm(zs[start] - zs[start - 1])) if start > 0 else 0.0
                event.pop("start_joint_pos")
            np.savez_compressed(
                result_dir / f"goal_{transition_mode}_trace.npz",
                joint_pos=q, joint_vel=v, torque=tau, z=zs, action=actions, goal_joint_mae=errors,
            )
            summary_path = result_dir / f"goal_{transition_mode}_summary.json"
            summary_path.write_text(json.dumps({
                "transition_mode": transition_mode,
                "model_folder": str(model_folder),
                "goal_json": str(goal_json_path),
                "goal_names": goal_names,
                "dt_seconds": float(env.dt),
                "goal_switch_interval": int(goal_switch_interval),
                "transition_steps": int(transition_steps) if transition_mode == "cem" else 0,
                "termination_reason": termination_reason,
                "video": str(video_path) if video_path is not None else None,
                "trace": str(result_dir / f"goal_{transition_mode}_trace.npz"),
                "events": switch_events,
            }, indent=2) + "\n")
            print(f"[INFO] Saved goal transition metrics: {summary_path}")
        finally:
            if planner_env is not None:
                planner_env.close()
            if renderer is not None:
                renderer.close()
    finally:
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO goal inference.")
    parser.add_argument("--model-folder", type=Path, default=default_model_folder(),
                        help="Trained run directory; defaults to the stable 23DoF demo checkpoint.")
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="goal inference")
    parser.add_argument("--goal-json", type=Path, default=None)
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save policy rollout MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--episode-len", type=int, default=2100)
    parser.add_argument("--goal-switch-interval", type=int, default=100)
    parser.add_argument("--goal-indices", type=int, nargs="+", default=None, help="Indices in generated goal_latents.npz to cycle during rollout.")
    parser.add_argument("--transition-mode", choices=("hard", "cem"), default="hard")
    parser.add_argument("--transition-steps", type=int, default=40, help="Steps to move from the old z to the new z.")
    parser.add_argument("--cem-candidates", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-knots", type=int, default=3)
    parser.add_argument("--cem-basis-dim", type=int, default=2)
    parser.add_argument("--goal-tolerance", type=float, default=0.25, help="Final mean absolute joint error in radians.")
    parser.add_argument("--transition-seed", type=int, default=0)
    parser.add_argument("--transition-output-dir", type=Path, default=None)
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
    run_goal_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        goal_json=args.goal_json,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        episode_len=args.episode_len,
        goal_switch_interval=args.goal_switch_interval,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
        goal_indices=args.goal_indices,
        transition_mode=args.transition_mode,
        transition_steps=args.transition_steps,
        cem_candidates=args.cem_candidates,
        cem_iterations=args.cem_iterations,
        cem_knots=args.cem_knots,
        cem_basis_dim=args.cem_basis_dim,
        goal_tolerance=args.goal_tolerance,
        transition_seed=args.transition_seed,
        transition_output_dir=args.transition_output_dir,
    )


if __name__ == "__main__":
    main()
