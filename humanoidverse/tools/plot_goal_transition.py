"""Plot recorded hard-switch and CEM goal-reaching traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(summary_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(summary_path.read_text())
    trace_path = Path(summary["trace"])
    if not trace_path.is_absolute():
        trace_path = summary_path.parent / trace_path
    with np.load(trace_path, allow_pickle=False) as source:
        trace = {key: source[key] for key in source.files}
    return summary, trace


def plot_comparison(hard_path: Path, cem_path: Path, output: Path) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    cem_summary = None
    for label, path, color in (("Hard switch", hard_path, "#b64949"), ("CEM path", cem_path, "#287a91")):
        summary, trace = _load(path)
        if label == "CEM path":
            cem_summary = summary
        dt = float(summary["dt_seconds"])
        times = np.arange(len(trace["joint_vel"])) * dt
        speed = np.sqrt(np.mean(np.square(trace["joint_vel"]), axis=1))
        torque = np.max(np.abs(trace["torque"]), axis=1)
        z = trace["z"]
        z_change = np.r_[0.0, np.linalg.norm(np.diff(z, axis=0), axis=1)]
        for ax, values in zip(axes, (speed, torque, trace["goal_joint_mae"], z_change)):
            ax.plot(times, values, label=label, color=color, lw=1.6)
        if summary.get("termination_reason"):
            axes[0].axvline(len(times) * dt, color=color, linestyle=":", alpha=0.8)
    assert cem_summary is not None
    for event in cem_summary["events"][1:]:
        time = int(event["step"]) * float(cem_summary["dt_seconds"])
        for ax in axes:
            ax.axvline(time, color="#777777", linestyle="--", alpha=0.45)
    axes[0].set_ylabel("Joint speed RMS (rad/s)")
    axes[1].set_ylabel("Peak |torque| (Nm)")
    axes[2].set_ylabel("Target joint MAE (rad)")
    axes[3].set_ylabel("Per-step z change (L2)")
    axes[3].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Goal transitions: hard switch vs sampled CEM path")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-summary", type=Path, required=True)
    parser.add_argument("--cem-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(plot_comparison(args.hard_summary, args.cem_summary, args.output))


if __name__ == "__main__":
    main()
