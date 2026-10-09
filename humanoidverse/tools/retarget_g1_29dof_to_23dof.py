"""Retarget G1 29-DoF RobotState CSV motions to the official G1 23-DoF model.

The 29-DoF pose is first projected by joint name.  A damped least-squares IK
step then adjusts the retained shoulder/elbow/wrist-roll joints so that the
23-DoF elbow and hand poses follow the 29-DoF forward-kinematics targets after
waist roll/pitch and wrist pitch/yaw have been removed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_TARGET_XML = str(Path(__file__).resolve().parents[2] / "humanoidverse/data/robots/g1_23dof/g1_23dof.xml")

BODY_PAIRS = (
    ("left_elbow_link", "left_elbow_link", False),
    ("left_wrist_roll_link", "left_wrist_roll_rubber_hand", True),
    ("right_elbow_link", "right_elbow_link", False),
    ("right_wrist_roll_link", "right_wrist_roll_rubber_hand", True),
)

IK_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


@dataclass
class MotionReport:
    name: str
    frames: int
    seconds: float
    clipped_values: int
    projected_position_rmse_m: float
    ik_position_rmse_m: float
    projected_wrist_orientation_rmse_rad: float
    ik_wrist_orientation_rmse_rad: float
    position_p95_m: float
    wrist_orientation_p95_rad: float


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    value = mujoco.mj_id2name(model, obj, int(idx))
    if value is None:
        raise ValueError(f"Unnamed MuJoCo object: type={obj}, id={idx}")
    return value


def _actuated_joints(model: mujoco.MjModel) -> list[str]:
    result: list[str] = []
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"Actuator {actuator_id} is not a joint transmission")
        result.append(_name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator_id, 0])))
    return result


def _joint_layout(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_addr: list[int] = []
    dof_addr: list[int] = []
    limits: list[list[float]] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} is missing from model")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Joint {joint_name!r} is not a hinge joint")
        qpos_addr.append(int(model.jnt_qposadr[joint_id]))
        dof_addr.append(int(model.jnt_dofadr[joint_id]))
        if bool(model.jnt_limited[joint_id]):
            limits.append([float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1])])
        else:
            limits.append([-np.pi, np.pi])
    return np.asarray(qpos_addr), np.asarray(dof_addr), np.asarray(limits)


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(current @ target.T).as_rotvec()


def _load_csv(path: Path, expected_columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise ValueError(f"{path} must have shape [T, {expected_columns}], got {values.shape}")
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} has too few frames or contains non-finite values")
    quat_norm = np.linalg.norm(values[:, 3:7], axis=1)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError(f"{path} contains a zero root quaternion")
    values[:, 3:7] /= quat_norm[:, None]
    return values


def _retarget_one(job: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(job["source_path"])
    output_path = Path(job["output_path"])
    source_model = mujoco.MjModel.from_xml_path(job["source_xml"])
    target_model = mujoco.MjModel.from_xml_path(job["target_xml"])
    source_data = mujoco.MjData(source_model)
    target_data = mujoco.MjData(target_model)

    source_joints = _actuated_joints(source_model)
    target_joints = _actuated_joints(target_model)
    source_qpos, _, _ = _joint_layout(source_model, source_joints)
    target_qpos, target_dof, target_limits = _joint_layout(target_model, target_joints)
    source_index = {name: idx for idx, name in enumerate(source_joints)}
    missing = [name for name in target_joints if name not in source_index]
    if missing:
        raise ValueError(f"23-DoF target joints missing from source model: {missing}")
    projection = np.asarray([source_index[name] for name in target_joints])
    ik_indices = np.asarray([target_joints.index(name) for name in IK_JOINTS])
    ik_qpos = target_qpos[ik_indices]
    ik_dof = target_dof[ik_indices]
    ik_limits = target_limits[ik_indices]

    source_body_ids = [mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, pair[0]) for pair in BODY_PAIRS]
    target_body_ids = [mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_BODY, pair[1]) for pair in BODY_PAIRS]
    if min(*source_body_ids, *target_body_ids) < 0:
        raise ValueError("An IK body pair is missing from a MuJoCo model")

    frames = _load_csv(source_path, 7 + len(source_joints))
    output_dof = np.empty((len(frames), len(target_joints)), dtype=np.float32)
    position_before: list[float] = []
    position_after: list[float] = []
    orientation_before: list[float] = []
    orientation_after: list[float] = []
    clipped_values = 0
    previous_ik: np.ndarray | None = None
    position_weight = float(job["position_weight"])
    orientation_weight = float(job["orientation_weight"])
    regularization = float(job["regularization"])
    temporal_regularization = float(job["temporal_regularization"])
    damping = float(job["damping"])
    max_delta = float(job["max_delta"])
    iterations = int(job["iterations"])
    sqrt_position_weight = np.sqrt(position_weight)
    sqrt_orientation_weight = np.sqrt(orientation_weight)
    eye = np.eye(len(ik_indices))
    jac_pos = np.empty((3, target_model.nv))
    jac_rot = np.empty((3, target_model.nv))
    started = time.perf_counter()

    for frame_idx, row in enumerate(frames):
        source_data.qpos[:3] = row[:3]
        source_data.qpos[3:7] = row[[6, 3, 4, 5]]
        source_data.qpos[source_qpos] = row[7:]
        mujoco.mj_forward(source_model, source_data)

        projected = row[7:][projection]
        clipped = np.clip(projected, target_limits[:, 0] + 1.0e-6, target_limits[:, 1] - 1.0e-6)
        clipped_values += int(np.count_nonzero(np.abs(clipped - projected) > 1.0e-9))
        target_data.qpos[:3] = row[:3]
        target_data.qpos[3:7] = row[[6, 3, 4, 5]]
        target_data.qpos[target_qpos] = clipped
        mujoco.mj_forward(target_model, target_data)

        desired_positions = [source_data.xpos[body_id].copy() for body_id in source_body_ids]
        desired_rotations = [source_data.xmat[body_id].reshape(3, 3).copy() for body_id in source_body_ids]

        def measure() -> tuple[float, float]:
            pos = np.concatenate(
                [target_data.xpos[body_id] - desired for body_id, desired in zip(target_body_ids, desired_positions)]
            )
            ori = np.concatenate(
                [
                    _orientation_error(target_data.xmat[body_id].reshape(3, 3), desired)
                    for body_id, desired, pair in zip(target_body_ids, desired_rotations, BODY_PAIRS)
                    if pair[2]
                ]
            )
            return float(np.sqrt(np.mean(pos * pos))), float(np.sqrt(np.mean(ori * ori)))

        pos_error, ori_error = measure()
        position_before.append(pos_error)
        orientation_before.append(ori_error)
        reference_ik = clipped[ik_indices]

        for _ in range(iterations):
            residuals: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            for target_body_id, desired_pos, desired_rot, pair in zip(
                target_body_ids, desired_positions, desired_rotations, BODY_PAIRS
            ):
                mujoco.mj_jacBody(target_model, target_data, jac_pos, jac_rot, target_body_id)
                residuals.append(sqrt_position_weight * (target_data.xpos[target_body_id] - desired_pos))
                jacobians.append(sqrt_position_weight * jac_pos[:, ik_dof])
                if pair[2] and orientation_weight > 0.0:
                    residuals.append(
                        sqrt_orientation_weight
                        * _orientation_error(target_data.xmat[target_body_id].reshape(3, 3), desired_rot)
                    )
                    jacobians.append(sqrt_orientation_weight * jac_rot[:, ik_dof])
            residual = np.concatenate(residuals)
            jacobian = np.vstack(jacobians)
            current_ik = target_data.qpos[ik_qpos]
            gradient = jacobian.T @ residual + regularization * (current_ik - reference_ik)
            normal_regularization = damping + regularization
            if previous_ik is not None and temporal_regularization > 0.0:
                gradient += temporal_regularization * (current_ik - previous_ik)
                normal_regularization += temporal_regularization
            delta = -np.linalg.solve(jacobian.T @ jacobian + normal_regularization * eye, gradient)
            target_data.qpos[ik_qpos] = np.clip(
                current_ik + np.clip(delta, -max_delta, max_delta), ik_limits[:, 0], ik_limits[:, 1]
            )
            mujoco.mj_forward(target_model, target_data)

        pos_error, ori_error = measure()
        position_after.append(pos_error)
        orientation_after.append(ori_error)
        output_dof[frame_idx] = target_data.qpos[target_qpos]
        previous_ik = target_data.qpos[ik_qpos].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            root_pos=frames[:, :3].astype(np.float32),
            root_quat=frames[:, 3:7].astype(np.float32),
            dof_pos=output_dof,
            joint_names=np.asarray(target_joints),
            fps=np.asarray(float(job["fps"]), dtype=np.float32),
            source_path=np.asarray(str(source_path)),
            source_robot=np.asarray("unitree_g1_29dof"),
            target_robot=np.asarray("unitree_g1_23dof"),
            retarget_method=np.asarray("29dof_name_projection_plus_constrained_dls_ik"),
        )
    temporary_path.replace(output_path)
    report = MotionReport(
        name=source_path.stem,
        frames=len(frames),
        seconds=time.perf_counter() - started,
        clipped_values=clipped_values,
        projected_position_rmse_m=float(np.mean(position_before)),
        ik_position_rmse_m=float(np.mean(position_after)),
        projected_wrist_orientation_rmse_rad=float(np.mean(orientation_before)),
        ik_wrist_orientation_rmse_rad=float(np.mean(orientation_after)),
        position_p95_m=float(np.percentile(position_after, 95)),
        wrist_orientation_p95_rad=float(np.percentile(orientation_after, 95)),
    )
    return asdict(report)


def _aggregate(reports: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    total_frames = sum(int(item["frames"]) for item in reports)

    def weighted(key: str) -> float:
        return sum(float(item[key]) * int(item["frames"]) for item in reports) / total_frames

    return {
        "method": "official G1 29DoF projection plus constrained 23DoF damped-least-squares IK",
        "source_xml": str(Path(args.source_xml).resolve()),
        "target_xml": str(Path(args.target_xml).resolve()),
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "fps": args.fps,
        "motion_count": len(reports),
        "frame_count": total_frames,
        "wall_seconds": elapsed,
        "throughput_fps": total_frames / elapsed,
        "clipped_values": sum(int(item["clipped_values"]) for item in reports),
        "weighted_projected_position_rmse_m": weighted("projected_position_rmse_m"),
        "weighted_ik_position_rmse_m": weighted("ik_position_rmse_m"),
        "weighted_projected_wrist_orientation_rmse_rad": weighted("projected_wrist_orientation_rmse_rad"),
        "weighted_ik_wrist_orientation_rmse_rad": weighted("ik_wrist_orientation_rmse_rad"),
        "parameters": {
            "iterations": args.iterations,
            "position_weight": args.position_weight,
            "orientation_weight": args.orientation_weight,
            "regularization": args.regularization,
            "temporal_regularization": args.temporal_regularization,
            "damping": args.damping,
            "max_delta": args.max_delta,
            "workers": args.workers,
        },
        "motions": sorted(reports, key=lambda item: item["name"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-xml", required=True, help="Source G1 29DoF MJCF; kept outside this 23DoF repository.")
    parser.add_argument("--target-xml", default=DEFAULT_TARGET_XML)
    parser.add_argument("--input-dir", required=True, help="Directory containing source RobotState CSV motions.")
    parser.add_argument("--output-dir", required=True, help="Directory for retargeted 23DoF NPZ motions.")
    parser.add_argument("--report", default=None, help="Quality report JSON; defaults next to --output-dir.")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.03)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--temporal-regularization", type=float, default=0.0002)
    parser.add_argument("--damping", type=float, default=0.0001)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.iterations <= 0 or args.fps <= 0.0:
        raise ValueError("workers, iterations, and fps must be positive")
    source_paths = sorted(Path(args.input_dir).expanduser().glob(args.pattern))
    if not source_paths:
        raise FileNotFoundError(f"No input CSV files matched {Path(args.input_dir) / args.pattern}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for source_path in source_paths:
        output_path = output_dir / f"{source_path.stem}.npz"
        if output_path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {output_path}. Use --force to rebuild the dataset.")
        jobs.append({**vars(args), "source_path": str(source_path), "output_path": str(output_path)})

    print(f"Retargeting {len(jobs)} motions with {args.workers} workers", flush=True)
    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_retarget_one, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            report = future.result()
            reports.append(report)
            print(
                f"[{completed}/{len(jobs)}] {report['name']}: frames={report['frames']} "
                f"pos={report['projected_position_rmse_m']:.5f}->{report['ik_position_rmse_m']:.5f}m "
                f"wrist_ori={report['projected_wrist_orientation_rmse_rad']:.5f}->"
                f"{report['ik_wrist_orientation_rmse_rad']:.5f}rad",
                flush=True,
            )
    aggregate = _aggregate(reports, args, time.perf_counter() - started)
    report_path = Path(args.report).expanduser() if args.report else output_dir.parent / "reports/retarget_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "motions"}, indent=2), flush=True)
    print(f"Wrote report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
