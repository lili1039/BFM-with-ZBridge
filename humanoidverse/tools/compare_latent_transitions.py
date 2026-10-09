"""Compare continuous hard/CEM inference runs and animate their latent paths.

The sphere view is a quotient projection of the unit-normalized latent vectors:
two orthogonal axes retain their exact coordinates, and all remaining axes are
collapsed into the positive third coordinate. It is not a lossless 3-D view.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont

media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


METRICS = (
    ("rms_joint_speed_rad_s", "Joint speed RMS (rad/s)"),
    ("peak_joint_speed_rad_s", "Peak joint speed (rad/s)"),
    ("mean_joint_travel_rad", "Mean joint travel (rad)"),
    ("rms_joint_torque_nm", "Joint torque RMS (Nm)"),
    ("peak_abs_joint_torque_nm", "Peak torque (Nm)"),
    ("first_step_z_jump_l2", "First-step latent jump (L2)"),
    ("peak_step_z_change_l2", "Peak per-step latent change (L2)"),
)


def load_run(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    prefix = "goal_" if "goal_names" in summary else "reward_sequence_"
    mode = summary.get("transition_mode", summary.get("mode"))
    suffix = "" if prefix == "goal_" else "_0"
    stem = f"{prefix}{mode}{suffix}"
    root = summary_path.parent
    with np.load(root / f"{stem}_trace.npz") as trace:
        arrays = {key: trace[key].copy() for key in trace.files}
    return {"summary": summary, "trace": arrays, "video": root / f"{stem}.mp4"}


def projection_axes(z_hard: np.ndarray, z_cem: np.ndarray, switch_step: int) -> tuple[np.ndarray, np.ndarray]:
    before = z_cem[max(0, switch_step - 1)].astype(np.float64)
    after = z_hard[min(switch_step, len(z_hard) - 1)].astype(np.float64)
    e1 = before / np.linalg.norm(before)
    orthogonal = after - np.dot(after, e1) * e1
    if np.linalg.norm(orthogonal) < 1e-8:
        axis = np.argmin(np.abs(e1))
        orthogonal = np.eye(len(e1))[axis] - e1[axis] * e1
    e2 = orthogonal / np.linalg.norm(orthogonal)
    return e1, e2


def project_to_sphere(z: np.ndarray, axes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Project unit z onto S2; the positive third component is residual norm."""
    z = np.asarray(z, dtype=np.float64)
    unit = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    x, y = unit @ axes[0], unit @ axes[1]
    third = np.sqrt(np.maximum(0.0, 1.0 - x * x - y * y))
    return np.stack((x, y, third), axis=-1)


def transition_metrics(run: dict, start: int, duration: int) -> dict:
    trace = run["trace"]
    end = min(start + duration, len(trace["z"]))
    result = {"start_step": start, "observed_steps": max(0, end - start), "complete_window": end - start == duration}
    if end <= start:
        return result
    velocity = trace["joint_vel"][start:end]
    torque = trace["torque"][start:end]
    q = trace["joint_pos"]
    initial = trace["initial_joint_pos"] if start == 0 and "initial_joint_pos" in trace else q[start - 1]
    z = trace["z"]
    dz = np.linalg.norm(np.diff(z[start - 1:end], axis=0), axis=-1)
    result.update(
        rms_joint_speed_rad_s=float(np.sqrt(np.mean(velocity**2))),
        peak_joint_speed_rad_s=float(np.max(np.abs(velocity))),
        mean_joint_travel_rad=float(np.mean(np.sum(np.abs(np.diff(np.vstack((initial, q[start:end])), axis=0)), axis=0))),
        rms_joint_torque_nm=float(np.sqrt(np.mean(torque**2))),
        mean_abs_joint_torque_nm=float(np.mean(np.abs(torque))),
        peak_abs_joint_torque_nm=float(np.max(np.abs(torque))),
        first_step_z_jump_l2=float(dz[0]),
        peak_step_z_change_l2=float(np.max(dz)),
        mean_step_z_change_l2=float(np.mean(dz)),
        latent_path_length_l2=float(np.sum(dz)),
    )
    if "goal_joint_mae" in trace:
        result["last_valid_goal_joint_mae_rad"] = float(trace["goal_joint_mae"][end - 1])
        result["goal_joint_mae_rad"] = result["last_valid_goal_joint_mae_rad"] if result["complete_window"] else None
    return result


def save_metrics(hard: dict, cem: dict, out: Path, duration: int) -> list[dict]:
    events = cem["summary"].get("events", cem["summary"].get("switches"))
    rows = []
    for event in events[1:]:
        step = int(event["step"])
        for mode, run in (("hard", hard), ("cem", cem)):
            rows.append({"switch_step": step, "from": event["from"], "to": event["to"], "mode": mode,
                         **transition_metrics(run, step, duration)})
    (out / "transition_metrics.json").write_text(json.dumps({
        "transition_window_steps": duration,
        "dt_seconds": cem["summary"]["dt_seconds"],
        "rows": rows,
        "termination": {mode: run["summary"]["termination_reason"] for mode, run in (("hard", hard), ("cem", cem))},
        "total_steps": {mode: len(run["trace"]["z"]) for mode, run in (("hard", hard), ("cem", cem))},
        "projection": "e1=pre-switch z, e2=orthogonal target component; residual norm is positive third sphere coordinate",
        "torque_animation": "per-step root mean square across controlled joint torques, in Nm; legend gives RMS over all complete transition windows",
    }, indent=2) + "\n")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "transition_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    steps = [int(event["step"]) for event in events[1:]]
    for ax, (key, label) in zip(axes.flat, METRICS):
        x = np.arange(len(steps))
        for mode, offset, color in (("hard", -0.19, "#dc4a4a"), ("cem", 0.19, "#3183cc")):
            matches = [next(row for row in rows if row["switch_step"] == step and row["mode"] == mode) for step in steps]
            values = [row.get(key, np.nan) if row["complete_window"] else np.nan for row in matches]
            ax.bar(x + offset, values, width=0.38, label="BFM-zero" if mode == "hard" else "CEM", color=color)
        ax.set_title(label)
        ax.set_xticks(x, [f"step {step}" for step in steps], rotation=15)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if any(not row["complete_window"] for row in rows):
            ax.text(0.99, 0.98, "missing bar = early termination", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)
    axes.flat[0].legend()
    for ax in axes.flat[len(METRICS):]:
        ax.set_visible(False)
    fig.suptitle(f"Transition metrics over {duration} policy steps ({duration * cem['summary']['dt_seconds']:.2f} s)")
    fig.savefig(out / "transition_metrics.png", dpi=150)
    plt.close(fig)
    return rows


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def compose_video(hard: dict, cem: dict, out: Path, fps: int) -> int:
    a, b = media.read_video(str(hard["video"])), media.read_video(str(cem["video"]))
    if len(a) != len(hard["trace"]["z"]) or len(b) != len(cem["trace"]["z"]):
        raise ValueError("Video and latent trace frame counts disagree")
    n = max(len(a), len(b))
    height, width = a.shape[1:3]
    header = 80
    with imageio.get_writer(out / "comparison.mp4", fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for i in range(n):
            canvas = Image.new("RGB", (2 * width, height + header), "#151922")
            draw = ImageDraw.Draw(canvas)
            title = "Goal reaching transition" if "goal_names" in cem["summary"] else "Reward inference transition"
            draw.text((16, 7), title, font=font(19), fill="white")
            draw.text((2 * width - 140, 11), f"{i / fps:.2f}s  #{i:03d}", font=font(13), fill="#dce6f3")
            for j, (frames, method, color) in enumerate(((a, "BFM-zero", "#ff7777"), (b, "CEM", "#70baff"))):
                frame = Image.fromarray(frames[min(i, len(frames) - 1)]).convert("RGB")
                if i >= len(frames):
                    frame = frame.point(lambda value: int(value * 0.4))
                canvas.paste(frame, (j * width, header))
                draw.text((j * width + 16, 48), f"{method}  {'ENDED' if i >= len(frames) else 'RUNNING'}",
                          font=font(17), fill=color)
            writer.append_data(np.asarray(canvas))
    return n


def animate_sphere(hard: dict, cem: dict, out: Path, fps: int, n: int) -> None:
    z_hard, z_cem = hard["trace"]["z"], cem["trace"]["z"]
    events = cem["summary"].get("events", cem["summary"].get("switches"))
    first_switch = int(events[1]["step"])
    axes = projection_axes(z_hard, z_cem, first_switch)
    xyz = (project_to_sphere(z_hard, axes), project_to_sphere(z_cem, axes))
    torque_rms = [np.sqrt(np.mean(np.square(run["trace"]["torque"]), axis=-1)) for run in (hard, cem)]
    fig = plt.figure(figsize=(10, 6), dpi=100, facecolor="#101823")
    panels = [fig.add_axes((0.04 + 0.49 * i, 0.27, 0.45, 0.53), projection="3d") for i in range(2)]
    time_axes = fig.add_axes((0.08, 0.08, 0.86, 0.16))
    u = np.linspace(0, 2 * np.pi, 19)
    v = np.linspace(0, np.pi, 11)
    for ax, title, color, path in zip(panels, ("BFM-zero", "CEM"), ("#ff7777", "#70baff"), xyz):
        ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                          np.outer(np.ones_like(u), np.cos(v)), color="#586776", alpha=0.25, linewidth=0.6)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, alpha=0.25, linewidth=1)
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05),
               xlabel="e1", ylabel="e2", zlabel="residual")
        ax.view_init(elev=28, azim=55)
        ax.set_title(title, color=color, pad=3)
        ax.tick_params(labelsize=7, colors="white")
        ax.set_facecolor("#101823")
    dynamic_lines = [ax.plot([], [], [], color=color, linewidth=3)[0]
                     for ax, color in zip(panels, ("#ff7777", "#70baff"))]
    markers = [ax.plot([], [], [], marker="o", markersize=8, color=color)[0]
               for ax, color in zip(panels, ("#ff7777", "#70baff"))]
    transition_steps = int(cem["summary"]["transition_steps"])
    for run, torque, color, label in zip((hard, cem), torque_rms, ("#ff7777", "#70baff"), ("BFM-zero", "CEM")):
        windows = [run["trace"]["torque"][int(event["step"]):int(event["step"]) + transition_steps]
                   for event in events[1:] if int(event["step"]) + transition_steps <= len(run["trace"]["torque"])]
        window_rms = float(np.sqrt(np.mean(np.concatenate(windows, axis=0)**2))) if windows else float("nan")
        time_axes.plot(np.arange(len(torque)) / fps, torque, color=color,
                       label=f"{label} (switch RMS {window_rms:.2f} Nm)", linewidth=1.4)
    for event in events[1:]:
        time_axes.axvspan(event["step"] / fps, (event["step"] + transition_steps) / fps,
                          color="#c0cad5", alpha=0.05)
        time_axes.axvline(event["step"] / fps, color="#c0cad5", alpha=0.4, linestyle="--")
    cursor = time_axes.axvline(0, color="white", linewidth=1.5)
    time_axes.set(xlim=(0, n / fps), xlabel="Time (s)", ylabel="Joint torque RMS (Nm)")
    time_axes.set_facecolor("#182331")
    time_axes.tick_params(colors="white", labelsize=8)
    time_axes.xaxis.label.set_color("white")
    time_axes.yaxis.label.set_color("white")
    time_axes.legend(loc="upper right", facecolor="#182331", labelcolor="white", fontsize=7)
    fig.text(0.5, 0.95, "Latent motion on a 3D quotient of the 256D sphere", ha="center", color="white", fontsize=13)
    with imageio.get_writer(out / "z_sphere.mp4", fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for frame in range(n):
            for path, line, marker in zip(xyz, dynamic_lines, markers):
                j = min(frame, len(path) - 1)
                line.set_data_3d(path[:j + 1, 0], path[:j + 1, 1], path[:j + 1, 2])
                marker.set_data_3d(path[j:j + 1, 0], path[j:j + 1, 1], path[j:j + 1, 2])
            cursor.set_xdata([frame / fps, frame / fps])
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-summary", type=Path, required=True)
    parser.add_argument("--cem-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hard, cem = load_run(args.hard_summary), load_run(args.cem_summary)
    if hard["summary"].get("goal_names", hard["summary"].get("tasks")) != cem["summary"].get("goal_names", cem["summary"].get("tasks")):
        raise ValueError("Hard and CEM task sequences differ")
    duration = int(cem["summary"]["transition_steps"])
    save_metrics(hard, cem, args.output_dir, duration)
    frames = compose_video(hard, cem, args.output_dir, args.fps)
    animate_sphere(hard, cem, args.output_dir, args.fps, frames)
    print(f"Saved {frames} synchronized frames to {args.output_dir}")


if __name__ == "__main__":
    main()
