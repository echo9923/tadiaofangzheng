from __future__ import annotations

import math

import numpy as np

from tower_sim.dataclasses import Command, CraneState, CraneStatic
from tower_sim.geometry import wrap_to_pi


def load_acceleration_scale(load_weight: float, load_capacity: float, min_acc_scale: float = 0.5) -> float:
    """Compute load-dependent acceleration scaling."""

    if load_capacity <= 0.0:
        return min_acc_scale
    load_ratio = max(0.0, load_weight / load_capacity)
    return max(min_acc_scale, 1.0 - 0.4 * load_ratio)


def _limit_axis_velocity(
    current_dot: float,
    command_dot: float,
    max_dot: float,
    max_acc: float,
    tau: float,
    dt: float,
) -> tuple[float, float]:
    tau_safe = max(tau, 1e-6)
    desired_dot = current_dot + dt / tau_safe * (command_dot - current_dot)
    desired_dot = float(np.clip(desired_dot, -max_dot, max_dot))
    next_dot = float(np.clip(desired_dot, current_dot - max_acc * dt, current_dot + max_acc * dt))
    next_dot = float(np.clip(next_dot, -max_dot, max_dot))
    ddot = (next_dot - current_dot) / dt
    return next_dot, ddot


def update_state(
    state: CraneState,
    static: CraneStatic,
    command: Command,
    dt: float,
    h_clearance: float = 2.0,
    h_min: float = 0.0,
    min_acc_scale: float = 0.5,
    emergency_brake_scale: float = 2.0,
    normal_brake_scale: float = 1.0,
) -> CraneState:
    """Advance one crane state by one kinematic time step."""

    scale = load_acceleration_scale(state.load_weight, static.load_capacity, min_acc_scale=min_acc_scale)
    theta_cmd = command.theta_dot_cmd
    r_cmd = command.r_dot_cmd
    h_cmd = command.h_dot_cmd
    if state.r <= static.min_radius and r_cmd < 0.0:
        r_cmd = 0.0
    elif state.r >= static.max_radius and r_cmd > 0.0:
        r_cmd = 0.0
    h_upper = static.tower_height - h_clearance
    if state.h <= h_min and h_cmd < 0.0:
        h_cmd = 0.0
    elif state.h >= h_upper and h_cmd > 0.0:
        h_cmd = 0.0
    if command.emergency_flag:
        theta_cmd = 0.0
        r_cmd = 0.0
        h_cmd = 0.0
        scale *= max(1.0, float(emergency_brake_scale))
    elif command.brake_flag:
        scale *= max(1.0, float(normal_brake_scale))
    theta_dot, theta_ddot = _limit_axis_velocity(
        state.theta_dot,
        theta_cmd,
        static.max_theta_dot,
        static.max_theta_acc * scale,
        static.response_tau,
        dt,
    )
    r_dot, r_ddot = _limit_axis_velocity(
        state.r_dot,
        r_cmd,
        static.max_r_dot,
        static.max_r_acc * scale,
        static.response_tau,
        dt,
    )
    h_dot, h_ddot = _limit_axis_velocity(
        state.h_dot,
        h_cmd,
        static.max_h_dot,
        static.max_h_acc * scale,
        static.response_tau,
        dt,
    )

    theta = wrap_to_pi(state.theta + theta_dot * dt)
    r = state.r + r_dot * dt
    h = state.h + h_dot * dt

    if r < static.min_radius:
        r = static.min_radius
        if r_dot < 0.0:
            r_dot = 0.0
            r_ddot = (r_dot - state.r_dot) / dt
    elif r > static.max_radius:
        r = static.max_radius
        if r_dot > 0.0:
            r_dot = 0.0
            r_ddot = (r_dot - state.r_dot) / dt

    if h < h_min:
        h = h_min
        if h_dot < 0.0:
            h_dot = 0.0
            h_ddot = (h_dot - state.h_dot) / dt
    elif h > h_upper:
        h = h_upper
        if h_dot > 0.0:
            h_dot = 0.0
            h_ddot = (h_dot - state.h_dot) / dt

    return CraneState(
        theta=theta,
        r=r,
        h=h,
        theta_dot=theta_dot,
        r_dot=r_dot,
        h_dot=h_dot,
        theta_ddot=theta_ddot,
        r_ddot=r_ddot,
        h_ddot=h_ddot,
        load_weight=state.load_weight,
        task_id=state.task_id,
        task_index=state.task_index,
        task_stage=state.task_stage,
    )


def stop_command() -> Command:
    """Return a zero-velocity braking command."""

    return Command(0.0, 0.0, 0.0, brake_flag=1, emergency_flag=0)
