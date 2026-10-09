"""CEM planning of a finite-duration goal-latent transition in MJLab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator
from torch.utils._pytree import tree_map


@dataclass(frozen=True)
class TransitionConfig:
    steps: int = 40
    candidates: int = 8
    iterations: int = 3
    knots: int = 3
    basis_dim: int = 2
    goal_tolerance: float = 0.25
    velocity_weight: float = 1.0
    peak_velocity_weight: float = 0.2
    travel_weight: float = 0.5
    torque_weight: float = 0.01
    goal_weight: float = 30.0
    seed: int = 0

    def validate(self, interval: int) -> None:
        if not 2 <= self.steps <= interval:
            raise ValueError("transition steps must be between 2 and goal-switch-interval")
        if self.candidates < 2 or self.iterations < 1 or self.knots < 1:
            raise ValueError("CEM requires at least two candidates, one iteration, and one knot")
        if self.basis_dim < 0 or self.goal_tolerance <= 0:
            raise ValueError("basis_dim must be nonnegative and goal_tolerance positive")
        if any(x < 0 for x in (self.velocity_weight, self.peak_velocity_weight, self.travel_weight, self.torque_weight, self.goal_weight)):
            raise ValueError("cost weights must be nonnegative")


def latent_basis(all_goals: np.ndarray, z_from: np.ndarray, z_to: np.ndarray, basis_dim: int) -> np.ndarray:
    """Small, reproducible set of directions outside the source/target plane."""
    z_dim = z_from.size
    if basis_dim == 0 or len(all_goals) < 3:
        return np.empty((0, z_dim), dtype=np.float32)
    samples = np.asarray(all_goals, dtype=np.float64)
    samples = samples - samples.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(samples, full_matrices=False)
    directions: list[np.ndarray] = []
    for raw in vt:
        v = raw.copy()
        for fixed in (z_from, z_to, *directions):
            fixed = np.asarray(fixed, dtype=np.float64)
            v -= np.dot(v, fixed) / max(np.dot(fixed, fixed), 1e-12) * fixed
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            directions.append(v / norm)
        if len(directions) == basis_dim:
            break
    return np.asarray(directions, dtype=np.float32).reshape(-1, z_dim)


def build_latent_path(
    params: np.ndarray,
    z_from: np.ndarray,
    z_to: np.ndarray,
    basis: np.ndarray,
    steps: int,
    *,
    project: bool,
) -> np.ndarray:
    """Smooth path with fixed endpoints and monotone progress; params are CEM variables."""
    z_from = np.asarray(z_from, dtype=np.float64).reshape(-1)
    z_to = np.asarray(z_to, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64).reshape(-1, z_from.size)
    knots = len(params) // (1 + len(basis))
    if knots < 1 or len(params) != knots * (1 + len(basis)) or steps < 2:
        raise ValueError("invalid latent path parameter dimensions")
    knot_times = np.linspace(0.0, 1.0, knots + 2)
    sample_times = np.linspace(0.0, 1.0, steps)
    progress_knots = np.sort(1.0 / (1.0 + np.exp(-np.clip(params[:knots], -12.0, 12.0))))
    progress = PchipInterpolator(knot_times, np.r_[0.0, progress_knots, 1.0])(sample_times)
    path = (1.0 - progress[:, None]) * z_from + progress[:, None] * z_to
    if len(basis):
        offsets = np.tanh(params[knots:].reshape(knots, len(basis))) * (0.15 * np.sqrt(z_from.size))
        offsets = CubicSpline(
            knot_times,
            np.vstack([np.zeros(len(basis)), offsets, np.zeros(len(basis))]),
            axis=0,
            bc_type="clamped",
        )(sample_times)
        path += offsets @ basis
    if project:
        norms = np.linalg.norm(path, axis=-1, keepdims=True)
        path = np.sqrt(z_from.size) * path / np.maximum(norms, 1e-8)
    path[0] = z_from
    path[-1] = z_to
    return path.astype(np.float32)


def rank_candidates(costs: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Enforce goal completion before comparing motion costs."""
    return np.lexsort((np.asarray(costs), ~np.asarray(feasible, dtype=bool)))


def _repeat_snapshot(core: Any, count: int) -> dict[str, torch.Tensor]:
    root = core.robot_root_states[0:1].detach()
    dof = torch.stack((core.dof_pos[0:1], core.dof_vel[0:1]), dim=-1).detach()
    return {"root_states": root.repeat(count, 1), "dof_states": dof.repeat(count, 1, 1)}


@torch.no_grad()
def _evaluate_paths(
    model: Any,
    live_env: Any,
    live_obs: dict[str, torch.Tensor],
    planner_env: Any,
    paths: np.ndarray,
    target_dof: torch.Tensor | None,
    interval: int,
    config: TransitionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(paths)
    live = live_env._env
    planner = planner_env._env
    planner_env.reset(to_numpy=False, target_states=_repeat_snapshot(live, count))
    # reset() clears actor history. Restore the live history and last action so
    # every candidate starts with the same actor observation as the real robot.
    for key, value in live.history_handler.history.items():
        planner.history_handler.history[key][:] = value[0:1].repeat(count, 1, 1)
    planner.actions[:] = live.actions[0:1]
    planner.last_actions[:] = live.last_actions[0:1]
    planner.mjlab_env.episode_length_buf[:] = live.mjlab_env.episode_length_buf[0]
    planner.episode_length_buf[:] = live.episode_length_buf[0]
    planner.motion_ids[:] = 0
    planner.motion_start_times[:] = live.motion_start_times[0]
    observation = tree_map(lambda x: x[0:1].repeat(count, *([1] * (x.ndim - 1))), live_obs)

    device = target_dof.device if target_dof is not None else planner.dof_pos.device
    z_paths = torch.as_tensor(paths, device=device, dtype=torch.float32)
    target = target_dof.reshape(1, -1) if target_dof is not None else None
    prev_q = planner.dof_pos.detach().clone()
    speed_sq = torch.zeros(count, device=device)
    peak_speed = torch.zeros(count, device=device)
    peak_torque = torch.zeros(count, device=device)
    travel = torch.zeros(count, device=device)
    failed = torch.zeros(count, device=device, dtype=torch.bool)
    for step in range(interval):
        z = z_paths[:, min(step, paths.shape[1] - 1)]
        action = model.act(observation, z, mean=True)
        observation, _, terminated, truncated, _ = planner_env.step(action, to_numpy=False)
        q = planner.dof_pos.detach().clone()
        velocity = planner.dof_vel.detach().clone()
        failed |= torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
        failed |= ~torch.isfinite(q).all(dim=-1) | ~torch.isfinite(velocity).all(dim=-1)
        safe_velocity = torch.nan_to_num(velocity, nan=1e3, posinf=1e3, neginf=-1e3)
        speed_sq += safe_velocity.square().mean(dim=-1)
        peak_speed = torch.maximum(peak_speed, safe_velocity.abs().amax(dim=-1))
        peak_torque = torch.maximum(peak_torque, torch.nan_to_num(planner.torques.abs(), nan=1e3).amax(dim=-1))
        travel += torch.nan_to_num((q - prev_q).abs().mean(dim=-1), nan=1e3)
        prev_q = q
    rms_speed = torch.sqrt(speed_sq / interval)
    smooth_cost = (
        config.velocity_weight * rms_speed
        + config.peak_velocity_weight * peak_speed
        + config.travel_weight * travel
        + config.torque_weight * peak_torque
    )
    if target is None:
        # Reward tasks have no fixed joint-angle target. Completion is encoded
        # by the path endpoint, while validity still excludes failed rollouts.
        feasible = ~failed
        cost = smooth_cost + failed.float() * 1e5
    else:
        goal_mae = (planner.dof_pos - target).abs().mean(dim=-1)
        feasible = (~failed) & (goal_mae <= config.goal_tolerance)
        # Treat completion as a constraint: among candidates that reach the goal,
        # choose the gentlest motion instead of rewarding unnecessary accuracy.
        cost = smooth_cost + (~feasible).float() * (1e4 + config.goal_weight * goal_mae) + failed.float() * 1e5
    metrics = {
        "rms_speed": rms_speed.cpu().numpy(),
        "peak_speed": peak_speed.cpu().numpy(),
        "joint_travel": travel.cpu().numpy(),
        "peak_torque": peak_torque.cpu().numpy(),
        "feasible": feasible.cpu().numpy(),
        "failed": failed.cpu().numpy(),
    }
    if target is not None:
        metrics["goal_mae"] = goal_mae.cpu().numpy()
    return cost.cpu().numpy(), metrics


@torch.no_grad()
def optimize_transition(
    model: Any,
    live_env: Any,
    live_obs: dict[str, torch.Tensor],
    planner_env: Any,
    z_from: np.ndarray,
    z_to: np.ndarray,
    all_goals: np.ndarray,
    target_dof: torch.Tensor | None,
    interval: int,
    config: TransitionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run CEM from the live robot state and return the lowest-cost path."""
    config.validate(interval)
    if planner_env.num_envs != config.candidates:
        raise ValueError("planner environment count must equal CEM candidates")
    basis = latent_basis(all_goals, z_from, z_to, config.basis_dim)
    knots = config.knots
    seed_progress = np.arange(1, knots + 1, dtype=np.float64) / (knots + 1)
    mean = np.r_[np.log(seed_progress / (1 - seed_progress)), np.zeros(knots * len(basis))]
    std = np.r_[np.full(knots, 0.9), np.full(knots * len(basis), 0.5)]
    rng = np.random.default_rng(config.seed)
    best_cost = float("inf")
    best_feasible = False
    best_path: np.ndarray | None = None
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, float]] = []
    seed_cost: float | None = None
    seed_metrics: dict[str, float] = {}
    for iteration in range(config.iterations):
        samples = rng.normal(mean, std, size=(config.candidates, len(mean)))
        samples[0] = mean
        if iteration == 0:
            samples[0] = np.r_[np.log(seed_progress / (1 - seed_progress)), np.zeros(knots * len(basis))]
            if config.candidates >= 3:
                for sample_idx, exponent in ((1, 0.6), (2, 1.6)):
                    progress = seed_progress ** exponent
                    samples[sample_idx] = np.r_[np.log(progress / (1 - progress)), np.zeros(knots * len(basis))]
        paths = np.stack(
            [build_latent_path(p, z_from, z_to, basis, config.steps, project=bool(model.cfg.archi.norm_z)) for p in samples]
        )
        costs, metrics = _evaluate_paths(model, live_env, live_obs, planner_env, paths, target_dof, interval, config)
        if iteration == 0:
            seed_cost = float(costs[0])
            seed_metrics = {key: float(value[0]) for key, value in metrics.items()}
        order = rank_candidates(costs, metrics["feasible"])
        elite = order[: max(2, config.candidates // 4)]
        mean = samples[elite].mean(axis=0)
        std = np.maximum(samples[elite].std(axis=0), 0.08)
        winner = int(order[0])
        iteration_result = {"iteration": iteration, "best_cost": float(costs[winner])}
        if target_dof is not None:
            iteration_result["best_goal_mae"] = float(metrics["goal_mae"][winner])
        history.append(iteration_result)
        winner_feasible = bool(metrics["feasible"][winner])
        if (winner_feasible and not best_feasible) or (winner_feasible == best_feasible and float(costs[winner]) < best_cost):
            best_cost = float(costs[winner])
            best_feasible = winner_feasible
            best_path = paths[winner].copy()
            best_metrics = {key: float(value[winner]) for key, value in metrics.items()}
    assert best_path is not None
    return best_path, {
        "cost": best_cost,
        "predicted": best_metrics,
        "seed_cost": seed_cost,
        "seed_predicted": seed_metrics,
        "iterations": history,
        "basis_dim": len(basis),
    }
