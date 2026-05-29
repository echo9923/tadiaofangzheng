from __future__ import annotations

from typing import Sequence

import numpy as np

from tower_sim.dataclasses import Command, CraneState, CraneStatic, LiftingTask
from tower_sim.geometry import wrap_to_pi

TASK_STAGES = (
    "move_to_pickup",
    "lower_to_pickup",
    "lift_load",
    "transport_to_dropoff",
    "lower_to_dropoff",
    "release_load",
    "idle_or_next_task",
)


def choose_active_task(tasks: Sequence[LiftingTask], state: CraneState, timestamp: float) -> LiftingTask | None:
    """Choose the current or next task for a crane."""

    if not tasks:
        return None
    for task in tasks:
        if task.task_id == state.task_id and timestamp >= task.start_time:
            return task
    for task in tasks:
        if timestamp >= task.start_time:
            return task
    return tasks[0]


def _near_angle(a: float, b: float, tol: float) -> bool:
    return abs(wrap_to_pi(a - b)) <= tol


def _near_xyz(state: CraneState, theta: float, r: float, h: float, tol: float) -> bool:
    return _near_angle(state.theta, theta, tol) and abs(state.r - r) <= tol and abs(state.h - h) <= tol


def advance_task_stage(state: CraneState, task: LiftingTask | None, stage_tolerance: float) -> CraneState:
    """Advance task stage using geometric tolerance only."""

    if task is None:
        state.task_stage = "idle_or_next_task"
        state.load_weight = 0.0
        return state
    if state.task_id != task.task_id:
        state.task_id = task.task_id
        state.task_stage = "move_to_pickup"
        state.load_weight = 0.0

    stage = state.task_stage
    if stage == "move_to_pickup" and _near_xyz(
        state, task.pickup_theta, task.pickup_r, task.transport_h, stage_tolerance
    ):
        state.task_stage = "lower_to_pickup"
    elif stage == "lower_to_pickup" and _near_xyz(
        state, task.pickup_theta, task.pickup_r, task.pickup_h, stage_tolerance
    ):
        state.task_stage = "lift_load"
        state.load_weight = task.load_weight
    elif stage == "lift_load" and abs(state.h - task.transport_h) <= stage_tolerance:
        state.task_stage = "transport_to_dropoff"
    elif stage == "transport_to_dropoff" and _near_xyz(
        state, task.dropoff_theta, task.dropoff_r, task.transport_h, stage_tolerance
    ):
        state.task_stage = "lower_to_dropoff"
    elif stage == "lower_to_dropoff" and _near_xyz(
        state, task.dropoff_theta, task.dropoff_r, task.dropoff_h, stage_tolerance
    ):
        state.task_stage = "release_load"
    elif stage == "release_load":
        state.task_stage = "idle_or_next_task"
        state.load_weight = 0.0
    return state


def target_for_stage(task: LiftingTask | None, state: CraneState) -> tuple[float, float, float]:
    """Return target theta/r/h for the current task stage."""

    if task is None:
        return state.theta, state.r, state.h
    stage = state.task_stage
    if stage == "move_to_pickup":
        return task.pickup_theta, task.pickup_r, task.transport_h
    if stage == "lower_to_pickup":
        return task.pickup_theta, task.pickup_r, task.pickup_h
    if stage == "lift_load":
        return task.pickup_theta, task.pickup_r, task.transport_h
    if stage == "transport_to_dropoff":
        return task.dropoff_theta, task.dropoff_r, task.transport_h
    if stage == "lower_to_dropoff":
        return task.dropoff_theta, task.dropoff_r, task.dropoff_h
    if stage == "release_load":
        return task.dropoff_theta, task.dropoff_r, task.dropoff_h
    return state.theta, state.r, state.h


def make_nominal_command(
    state: CraneState,
    static: CraneStatic,
    target: tuple[float, float, float],
    k_theta: float,
    k_r: float,
    k_h: float,
) -> Command:
    """Generate clipped target velocity command from target state and current state."""

    theta_target, r_target, h_target = target
    theta_cmd = float(
        np.clip(k_theta * wrap_to_pi(theta_target - state.theta), -static.max_theta_dot, static.max_theta_dot)
    )
    r_cmd = float(np.clip(k_r * (r_target - state.r), -static.max_r_dot, static.max_r_dot))
    h_cmd = float(np.clip(k_h * (h_target - state.h), -static.max_h_dot, static.max_h_dot))
    return Command(theta_cmd, r_cmd, h_cmd, brake_flag=0, emergency_flag=0)


def smooth_command(current: Command, previous: Command | None, smoothing: float) -> Command:
    """Blend the current command with the previous one."""

    if previous is None or smoothing <= 0.0:
        return current
    alpha = float(np.clip(smoothing, 0.0, 1.0))
    return Command(
        theta_dot_cmd=(1.0 - alpha) * current.theta_dot_cmd + alpha * previous.theta_dot_cmd,
        r_dot_cmd=(1.0 - alpha) * current.r_dot_cmd + alpha * previous.r_dot_cmd,
        h_dot_cmd=(1.0 - alpha) * current.h_dot_cmd + alpha * previous.h_dot_cmd,
        brake_flag=max(current.brake_flag, previous.brake_flag),
        emergency_flag=max(current.emergency_flag, previous.emergency_flag),
    )
