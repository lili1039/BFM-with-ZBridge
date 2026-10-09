"""Generate a portable G1 23DoF FB inference bundle from a completed run.

Example: uv run python -m humanoidverse.tools.export_wsl_bundle --checkpoint /data/UFO/runs/my_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from humanoidverse.utils.motion_data.paths import expand_motion_paths

from humanoidverse.tools.stage_continuation import resolve_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/data/lafan_g1_23dof_ik.yaml"
DEFAULT_GOALS = PROJECT_ROOT / "configs/goals/lafan_g1_23dof_ik.json"
DEFAULT_MOTION_IDS = (2, 9, 17, 22, 28)  # dance, get-up, fight, run, walk in the 40 full motions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_latents(path: Path, z_dim: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["names"])
        latents = np.asarray(data["latents"], dtype=np.float32)
    if names.ndim != 1 or latents.shape != (len(names), z_dim):
        raise ValueError(f"Invalid named latents in {path}: names={names.shape}, z={latents.shape}")
    if len(names) == 0 or len(set(names.tolist())) != len(names) or not np.isfinite(latents).all():
        raise ValueError(f"Empty, duplicate, or nonfinite named latents in {path}")
    return names, latents


def _tracking_latents(path: Path, motion_id: int, z_dim: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=np.float32)
        stored_id = int(data["motion_id"])
    if stored_id != motion_id or z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != z_dim or not np.isfinite(z).all():
        raise ValueError(f"Invalid tracking latents in {path}: motion_id={stored_id}, z={z.shape}")
    return z


def _read_run(checkpoint_arg: Path) -> tuple[Path, Path, dict, dict]:
    checkpoint, run_dir = resolve_checkpoint(checkpoint_arg)
    run_config_path = run_dir / "config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing training configuration next to checkpoint: {run_config_path}")
    with run_config_path.open() as file:
        run_config = json.load(file)
    with (checkpoint / "train_status.json").open() as file:
        status = json.load(file)
    return checkpoint, run_dir, run_config, status


def _robot_config(run_config: dict, override: Path | None) -> Path:
    raw = override or run_config.get("env", {}).get("robot_config_path")
    if not raw:
        raise ValueError("Could not infer robot config; provide --robot-config")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _motion_names(manifest: Path, dataset: str) -> list[str]:
    import yaml

    manifest_data = yaml.safe_load(manifest.read_text())
    matches = [entry for entry in manifest_data["datasets"] if entry["name"] == dataset]
    if len(matches) != 1 or matches[0].get("format") != "robot_state_npz":
        raise ValueError(f"Expected one robot_state_npz dataset named {dataset} in {manifest}")
    source = matches[0]["source_path"]
    names = [path.stem for path in expand_motion_paths(source, base_dir=manifest.parent, suffix=".npz")]
    return names


def _control_contract(robot_config: Path, onnx_meta: dict, step: int, robot_dst: Path) -> dict:
    import mujoco
    import yaml

    spec = yaml.safe_load(robot_config.read_text())
    if spec.get("name") != "g1_23dof":
        raise ValueError("This exporter currently supports the trained G1 23DoF configuration only")
    training = spec["training"]
    control = training["control"]
    joint_names = list(spec["control_joints"]["names"])
    if joint_names != onnx_meta["control_joint_names"]:
        raise ValueError("Robot joint order differs from policy ONNX metadata")
    xml_name = Path(spec["xml_path"]).name
    model = mujoco.MjModel.from_xml_path(str(robot_dst / xml_name))
    if int(model.nu) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} XML motors, found {model.nu}")

    def joint_value(name: str, mapping: dict) -> float:
        return next(float(value) for key, value in mapping.items() if key in name)

    joints = []
    for index, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Robot XML lacks joint {name}")
        actuator_ids = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == joint_id]
        if len(actuator_ids) != 1:
            raise ValueError(f"Expected one gear=1 actuator for {name}; found {actuator_ids}")
        actuator_id = actuator_ids[0]
        if float(model.actuator_gear[actuator_id, 0]) != 1.0:
            raise ValueError(f"Expected gear=1 actuator for {name}")
        kp = joint_value(name, control["stiffness"])
        effort = float(control["effort_limit"][index])
        joints.append(
            {
                "policy_index": index,
                "name": name,
                "mujoco_qpos_adr": int(model.jnt_qposadr[joint_id]),
                "mujoco_qvel_adr": int(model.jnt_dofadr[joint_id]),
                "mujoco_ctrl_adr": actuator_id,
                "default_dof_pos": float(training["init_state"]["default_joint_angles"][name]),
                "kp": kp,
                "kd": joint_value(name, control["damping"]),
                "effort_limit": effort,
                "action_target_scale": float(control["action_scale"]) * effort / kp,
            }
        )
    return {
        "checkpoint_global_steps": step,
        "robot": "g1_23dof",
        "xml": f"robot/{xml_name}",
        "policy_onnx": "model/FBcprAuxModel.onnx",
        "simulation_hz": 200,
        "policy_hz": 50,
        "decimation": 4,
        "actor_obs_order": ["state", "last_action", "history_actor", "z"],
        "state_order": ["dof_pos_relative[23]", "dof_vel[23]", "projected_gravity_body[3]", "base_ang_vel_body_times_0.25[3]"],
        "history_group_order": ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
        "history_frames_per_group": 4,
        "history_frame_order": "newest_first; current frame pushed after observation is built",
        "actor_obs_dim": int(onnx_meta["actor_obs_dim"]),
        "z_dim": int(onnx_meta["z_dim"]),
        "action_formula": "a=clip(raw_action*5,-5,5); target=default_dof_pos+a*action_target_scale; tau=clip(kp*(target-q)-kd*qd,-effort_limit,effort_limit); data.ctrl[mujoco_ctrl_adr]=tau",
        "root_qpos_quaternion_order": "wxyz",
        "joints": joints,
    }


def _commands(work: Path, robot: Path, manifest: Path, dataset: str, goals: Path, args: argparse.Namespace) -> list[list[str]]:
    common = [
        "--model-folder", str(work),
        "--data-manifest", str(manifest),
        "--dataset", dataset,
        "--robot-config", str(robot),
        "--device", args.device,
        "--headless", "true",
        "--save-mp4", "false",
        "--disable-dr", "true",
        "--disable-obs-noise", "true",
    ]
    tracking = [
        sys.executable, "-m", "humanoidverse.tracking_inference", *common,
        "--motion-list", *[str(i) for i in args.tracking_motion_ids],
        "--latents-only", "--export-onnx", "true",
    ]
    goal = [
        sys.executable, "-m", "humanoidverse.goal_inference", *common,
        "--goal-json", str(goals), "--export-onnx", "false",
    ]
    reward = [
        sys.executable, "-m", "humanoidverse.reward_inference", *common,
        "--num-samples", str(args.reward_samples),
        "--n-inferences", "1", "--skip-rollouts", "true",
        "--max-workers", str(args.max_workers), "--export-onnx", "false",
    ]
    if args.reward_tasks:
        reward.extend(["--tasks", *args.reward_tasks])
    # Reward inference exercises the largest dependency (the replay buffer),
    # so fail there before spending time on goal/tracking and ONNX export.
    return [reward, goal, tracking]


def _write_readme(bundle: Path, step: int, reward_names: list[str], goal_names: list[str], tracking_lengths: dict[int, int], motion_names: list[str], include_safetensors: bool) -> None:
    tracking_lines = "\n".join(f"- Motion ID `{i}` (`{motion_names[i]}`): `{length}` policy steps, `latents/tracking/zs_{i}.npz`" for i, length in tracking_lengths.items())
    text = f"""# UFO G1 23DoF FB: WSL / MuJoCo inference bundle

Checkpoint: `{step:,}` global environment transitions. All model files and latent arrays in this directory come from this checkpoint. This bundle contains inference assets; it does not include the replay buffer or optimizer state needed for resumed training.

## Contents

- `model/FBcprAuxModel.onnx`: actor policy, `actor_obs[batch,631] -> action[batch,23]`.
- `model/backward_encoder.onnx`: backward encoder for goal/tracking experiments.
- `model/control_contract.json`: joint order, MuJoCo indices, observation layout, policy timing, action scaling and PD gains.
- `latents/tasks.npz`: {len(reward_names)} reward and {len(goal_names)} goal vectors, each 256-dimensional float32.
- `latents/tracking/*.npz`: one time-varying z array per selected motion, at 50 Hz.
- `robot/`: portable G1 23DoF XML and meshes.
- `config/`: robot and goal definitions used to generate the latents.
- `manifest.json` and `SHA256SUMS`: provenance and integrity checks.
"""
    if include_safetensors:
        text += "- `checkpoint/`: complete inference safetensors and model configuration (no replay buffer or optimizer).\n"
    text += f"""
## Available task vectors

Reward: {', '.join(f'`{name}`' for name in reward_names)}.

Goal: {', '.join(f'`{name}`' for name in goal_names)}.

Tracking:\n{tracking_lines}

## Load a z and run the ONNX actor

```python
import numpy as np
import onnxruntime as ort

with np.load('latents/tasks.npz', allow_pickle=False) as data:
    names = data['reward_names'].tolist()
    z = data['reward_latents'][names.index('{reward_names[0]}')][None, :]

# Build state[52], last_action[23], history_actor[300] from MuJoCo state.
actor_obs = np.concatenate([state, last_action, history_actor, z], axis=1).astype(np.float32)
session = ort.InferenceSession('model/FBcprAuxModel.onnx', providers=['CPUExecutionProvider'])
raw_action = session.run(['action'], {{'actor_obs': actor_obs}})[0]
```

`z` occupies `actor_obs[:,375:631]`. A reward or goal vector stays fixed until you switch tasks. For tracking, load `latents/tracking/zs_<motion_id>.npz` and use `data['z'][step]` at each 50 Hz policy step. The tracking array starts from the first policy transition after the reference initial frame. Do not treat an isolated goal vector as a full reference trajectory.

The history observation is grouped by **field**, each with four newest-first frames: actions, body angular velocity, relative joint positions, joint velocities, projected gravity. It is not interleaved frame by frame. The policy output is normalized action, not motor torque; apply `model/control_contract.json` before writing MuJoCo `data.ctrl`. MuJoCo root quaternion order is `wxyz`.

Verify files after transfer with `sha256sum -c SHA256SUMS` from this directory. The JSON manifest records the exact checkpoint step and per-file hashes. The model XML and meshes are included, but the closed-loop MuJoCo runner on WSL must implement the observation/history and PD control contract.
"""
    (bundle / "README.md").write_text(text)


def build_bundle(args: argparse.Namespace) -> tuple[Path, Path | None]:
    checkpoint, run_dir, run_config, status = _read_run(args.checkpoint)
    robot = _robot_config(run_config, args.robot_config)
    manifest = args.data_manifest.expanduser().resolve(strict=True)
    goals = args.goal_json.expanduser().resolve(strict=True)
    import yaml

    robot_spec = yaml.safe_load(robot.read_text())
    if robot_spec.get("name") != "g1_23dof":
        raise ValueError("The default task and motion definitions apply only to G1 23DoF")
    source_xml = Path(robot_spec["xml_path"]).expanduser().resolve(strict=True)
    if not source_xml.is_file():
        raise FileNotFoundError(source_xml)
    step = int(status.get("global_time", status.get("time", 0)))
    output = args.output_dir or run_dir.parent.parent / "exports" / f"{run_dir.name}_step{step}_wsl"
    output = output.expanduser().absolute()
    archive = Path(f"{output}.tar.gz") if args.archive else None
    if output.exists() or (archive is not None and archive.exists()):
        raise FileExistsError(f"Refusing to overwrite existing bundle or archive: {output}")
    if run_dir == output or run_dir in output.parents or output in run_dir.parents:
        raise ValueError("Output directory must be separate from the training run")
    if len(set(args.tracking_motion_ids)) != len(args.tracking_motion_ids) or not args.tracking_motion_ids:
        raise ValueError("--tracking-motion-ids must contain unique IDs")
    motion_names = _motion_names(manifest, args.dataset)
    invalid_ids = [i for i in args.tracking_motion_ids if i < 0 or i >= len(motion_names)]
    if invalid_ids:
        raise ValueError(f"Tracking motion IDs {invalid_ids} are outside the {len(motion_names)} full motions in {manifest}")

    if args.dry_run:
        print(f"Checkpoint: {checkpoint}\nStep: {step}\nRobot: {robot}\nManifest: {manifest}\nGoals: {goals}\nOutput: {output}")
        for command in _commands(output / ".inference", robot, manifest, args.dataset, goals, args):
            print("Would run:", " ".join(command))
        return output, archive

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as scratch:
        scratch = Path(scratch)
        work = scratch / "inference"
        work.mkdir()
        (work / "checkpoint").symlink_to(checkpoint, target_is_directory=True)
        shutil.copy2(run_dir / "config.json", work / "config.json")
        bundle = scratch / output.name
        bundle.mkdir()
        env = os.environ.copy()
        if "UFO_CACHE_DIR" not in env and run_dir.parent.name == "runs":
            env["UFO_CACHE_DIR"] = str(run_dir.parent.parent / "cache")
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        for command in _commands(work, robot, manifest, args.dataset, goals, args):
            print("[bundle] Running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

        for folder in ("model", "robot", "config", "latents/tracking"):
            (bundle / folder).mkdir(parents=True, exist_ok=True)
        exported = work / "exported"
        shutil.copy2(exported / "FBcprAuxModel.onnx", bundle / "model/FBcprAuxModel.onnx")
        shutil.copy2(exported / "backward_encoder.onnx", bundle / "model/backward_encoder.onnx")
        with (exported / "FBcprAuxModel.meta.json").open() as file:
            onnx_meta = json.load(file)
        z_dim = int(onnx_meta["z_dim"])
        reward_names, reward_z = _named_latents(work / "reward_inference/reward_latents.npz", z_dim)
        goal_names, goal_z = _named_latents(work / "goal_inference/goal_latents.npz", z_dim)
        tracking_lengths = {}
        for motion_id in args.tracking_motion_ids:
            src = work / "tracking_inference" / f"zs_{motion_id}.npz"
            z = _tracking_latents(src, motion_id, z_dim)
            tracking_lengths[motion_id] = len(z)
            shutil.copy2(src, bundle / "latents/tracking" / src.name)
        np.savez_compressed(
            bundle / "latents/tasks.npz",
            reward_names=reward_names,
            reward_latents=reward_z,
            goal_names=goal_names,
            goal_latents=goal_z,
            checkpoint_global_steps=np.int64(step),
            latent_dim=np.int64(z_dim),
        )

        shutil.copytree(source_xml.parent, bundle / "robot", dirs_exist_ok=True)
        shutil.copy2(robot, bundle / "config/robot.yaml")
        shutil.copy2(goals, bundle / "config/goals.json")
        shutil.copy2(manifest, bundle / "config/data_manifest.yaml")
        onnx_meta["robot_config_path"] = "config/robot.yaml"
        onnx_meta["xml_path"] = f"robot/{source_xml.name}"
        (bundle / "model/FBcprAuxModel.meta.json").write_text(json.dumps(onnx_meta, indent=2) + "\n")
        contract = _control_contract(robot, onnx_meta, step, bundle / "robot")
        (bundle / "model/control_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        if args.include_safetensors:
            (bundle / "checkpoint/model").mkdir(parents=True)
            for name in ("config.json", "init_kwargs.json", "train_status.json"):
                shutil.copy2(checkpoint / name, bundle / "checkpoint" / name)
            for name in ("config.json", "init_kwargs.json", "model.safetensors"):
                shutil.copy2(checkpoint / "model" / name, bundle / "checkpoint/model" / name)
        _write_readme(bundle, step, reward_names.tolist(), goal_names.tolist(), tracking_lengths, motion_names, args.include_safetensors)

        artifacts = {
            str(file.relative_to(bundle)): {"bytes": file.stat().st_size, "sha256": _sha256(file)}
            for file in sorted(bundle.rglob("*"))
            if file.is_file()
        }
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_checkpoint": str(checkpoint),
                    "global_env_steps": step,
                    "optimizer_steps": int(status.get("optimizer_steps", 0)),
                    "robot": "g1_23dof",
                    "reward_names": reward_names.tolist(),
                    "goal_names": goal_names.tolist(),
                    "tracking_lengths": tracking_lengths,
                    "tracking_motion_names": {i: motion_names[i] for i in tracking_lengths},
                    "files": artifacts,
                },
                indent=2,
            )
            + "\n"
        )
        checksums = [f"{_sha256(file)}  {file.relative_to(bundle)}" for file in sorted(bundle.rglob("*")) if file.is_file()]
        (bundle / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        archive_tmp = None
        if archive is not None:
            archive_tmp = scratch / archive.name
            with tarfile.open(archive_tmp, "w:gz", compresslevel=1) as tar:
                tar.add(bundle, arcname=output.name)
        output.mkdir(exist_ok=False)
        os.rename(bundle, output)
        if archive is not None:
            os.rename(archive_tmp, archive)
    print(f"[bundle] Ready: {output}")
    if archive is not None:
        print(f"[bundle] Archive: {archive}")
    return output, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Completed training run or its checkpoint directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="New bundle directory; default is data/UFO/exports/<run>_step<N>_wsl")
    parser.add_argument("--robot-config", type=Path, default=None, help="Defaults to run/config.json env.robot_config_path")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", default="lafan_g1_23dof_ik")
    parser.add_argument("--goal-json", type=Path, default=DEFAULT_GOALS)
    parser.add_argument("--tracking-motion-ids", type=int, nargs="+", default=list(DEFAULT_MOTION_IDS))
    parser.add_argument("--reward-tasks", nargs="+", default=None, help="Default is all supported G1 23DoF locomotion tasks")
    parser.add_argument("--reward-samples", type=int, default=150000)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-safetensors", action="store_false", dest="include_safetensors", help="Create a smaller ONNX-only package")
    parser.add_argument("--no-archive", action="store_false", dest="archive")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned commands without generating files")
    args = parser.parse_args()
    if args.reward_samples <= 0 or args.max_workers <= 0:
        parser.error("--reward-samples and --max-workers must be positive")
    build_bundle(args)


if __name__ == "__main__":
    main()
