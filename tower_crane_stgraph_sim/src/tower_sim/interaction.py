from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from tower_sim.dataclasses import Command, CraneGeometry, CraneState, CraneStatic
from tower_sim.geometry import (
    constant_velocity_extrapolate,
    point_point_distance,
    reconstruct_geometry,
    segment_point_distance,
    segment_segment_distance,
    wrap_to_pi,
)
from tower_sim.layout import overlap_ratio


def _arm_hook_distance(g_i: CraneGeometry, g_j: CraneGeometry) -> float:
    return segment_point_distance(g_i.root, g_i.tip, g_j.hook)


def compute_pairwise_edge(
    static_i: CraneStatic,
    static_j: CraneStatic,
    state_i: CraneState,
    state_j: CraneState,
    geom_i: CraneGeometry,
    geom_j: CraneGeometry,
    prev_distances: dict[str, float] | None = None,
    dt: float = 1.0,
    thresholds: dict[str, float] | None = None,
) -> dict[str, float | int]:
    """Compute physical-prior edge features for one ordered crane pair."""

    d_arm_arm = segment_segment_distance(geom_i.root, geom_i.tip, geom_j.root, geom_j.tip)
    d_arm_hook_i_to_j = _arm_hook_distance(geom_i, geom_j)
    d_arm_hook_j_to_i = _arm_hook_distance(geom_j, geom_i)
    d_hook_hook = point_point_distance(geom_i.hook, geom_j.hook)
    d_arm_hook_min = min(d_arm_hook_i_to_j, d_arm_hook_j_to_i)
    dt_safe = max(dt, 1e-6)
    next_i = constant_velocity_extrapolate(state_i, dt_safe)
    next_j = constant_velocity_extrapolate(state_j, dt_safe)
    next_geom_i = reconstruct_geometry(static_i, next_i)
    next_geom_j = reconstruct_geometry(static_j, next_j)
    next_d_arm_arm = segment_segment_distance(next_geom_i.root, next_geom_i.tip, next_geom_j.root, next_geom_j.tip)
    next_d_arm_hook_min = min(_arm_hook_distance(next_geom_i, next_geom_j), _arm_hook_distance(next_geom_j, next_geom_i))
    next_d_hook_hook = point_point_distance(next_geom_i.hook, next_geom_j.hook)
    relative_approach_speed_arm_arm = max(0.0, (d_arm_arm - next_d_arm_arm) / dt_safe)
    relative_approach_speed_arm_hook = max(0.0, (d_arm_hook_min - next_d_arm_hook_min) / dt_safe)
    relative_approach_speed_hook_hook = max(0.0, (d_hook_hook - next_d_hook_hook) / dt_safe)
    relative_approach_speed = max(
        relative_approach_speed_arm_arm,
        relative_approach_speed_arm_hook,
        relative_approach_speed_hook_hook,
    )

    def ttc(distance: float, threshold: float, approach_speed: float) -> float:
        if distance <= threshold:
            return 0.0
        if approach_speed <= 1e-9:
            return -1.0
        return float((distance - threshold) / approach_speed)

    thresholds = thresholds or {}
    same_height = abs((static_i.base_z + static_i.tower_height) - (static_j.base_z + static_j.tower_height)) < max(
        thresholds.get("d_safe_arm_arm_m", 2.0), 1.0
    )
    base_distance = math.hypot(static_i.base_x - static_j.base_x, static_i.base_y - static_j.base_y)
    return {
        "crane_i": static_i.crane_id,
        "crane_j": static_j.crane_id,
        "crane_i_index": static_i.crane_index,
        "crane_j_index": static_j.crane_index,
        "d_arm_arm": d_arm_arm,
        "d_arm_hook_i_to_j": d_arm_hook_i_to_j,
        "d_arm_hook_j_to_i": d_arm_hook_j_to_i,
        "d_hook_hook": d_hook_hook,
        "delta_theta": wrap_to_pi(state_j.theta - state_i.theta),
        "delta_theta_dot": state_j.theta_dot - state_i.theta_dot,
        "delta_r": state_j.r - state_i.r,
        "delta_h": state_j.h - state_i.h,
        "delta_tower_height": static_j.tower_height - static_i.tower_height,
        "base_distance": base_distance,
        "overlap_ratio": overlap_ratio(static_i, static_j),
        "relative_approach_speed": relative_approach_speed,
        "relative_approach_speed_arm_arm": relative_approach_speed_arm_arm,
        "relative_approach_speed_arm_hook": relative_approach_speed_arm_hook,
        "relative_approach_speed_hook_hook": relative_approach_speed_hook_hook,
        "ttc_est_arm_arm": ttc(
            d_arm_arm,
            float(thresholds.get("d_safe_arm_arm_m", static_i.safety_radius_arm)),
            relative_approach_speed_arm_arm,
        ),
        "ttc_est_arm_hook": ttc(
            d_arm_hook_min,
            float(thresholds.get("d_safe_arm_hook_m", static_i.safety_radius_hook)),
            relative_approach_speed_arm_hook,
        ),
        "ttc_est_hook_hook": ttc(
            d_hook_hook,
            float(thresholds.get("d_safe_hook_hook_m", static_i.safety_radius_hook)),
            relative_approach_speed_hook_hook,
        ),
        "same_height_risk_zone": int(same_height and overlap_ratio(static_i, static_j) > 0.0),
    }


def compute_edges_for_step(
    statics: list[CraneStatic],
    states: dict[int, CraneState],
    geometries: dict[int, CraneGeometry],
    prev_pair_distances: dict[tuple[int, int], dict[str, float]],
    dt: float,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, float | int]], dict[tuple[int, int], dict[str, float]]]:
    """Compute all ordered pair edges for a simulation step."""

    rows: list[dict[str, float | int]] = []
    next_prev: dict[tuple[int, int], dict[str, float]] = {}
    static_by_id = {c.crane_index: c for c in statics}
    for i, j in itertools.permutations(sorted(static_by_id), 2):
        prev = prev_pair_distances.get((i, j))
        row = compute_pairwise_edge(
            static_by_id[i],
            static_by_id[j],
            states[i],
            states[j],
            geometries[i],
            geometries[j],
            prev_distances=prev,
            dt=dt,
            thresholds=thresholds,
        )
        rows.append(row)
        next_prev[(i, j)] = {
            "d_arm_arm": float(row["d_arm_arm"]),
            "d_arm_hook": min(float(row["d_arm_hook_i_to_j"]), float(row["d_arm_hook_j_to_i"])),
            "d_hook_hook": float(row["d_hook_hook"]),
        }
    return rows, next_prev


def classify_short_horizon_risk(
    statics: list[CraneStatic],
    states: dict[int, CraneState],
    horizon_s: float,
    dt: float,
    thresholds: dict[str, float],
) -> dict[int, str]:
    """Return yielding crane ids and the dominant future risk type."""

    risk_by_crane: dict[int, str] = {}
    static_by_id = {c.crane_index: c for c in statics}
    priority = {"arm_arm": 3, "arm_hook": 2, "hook_hook": 1}
    steps = max(1, int(math.ceil(horizon_s / max(dt, 1e-6))))
    for i, j in itertools.combinations(sorted(static_by_id), 2):
        pair_risk: str | None = None
        pair_priority = 0
        for k in range(1, steps + 1):
            tau = min(horizon_s, k * dt)
            state_i = constant_velocity_extrapolate(states[i], tau)
            state_j = constant_velocity_extrapolate(states[j], tau)
            geom_i = reconstruct_geometry(static_by_id[i], state_i)
            geom_j = reconstruct_geometry(static_by_id[j], state_j)
            d_arm_arm = segment_segment_distance(geom_i.root, geom_i.tip, geom_j.root, geom_j.tip)
            d_arm_hook = min(
                segment_point_distance(geom_i.root, geom_i.tip, geom_j.hook),
                segment_point_distance(geom_j.root, geom_j.tip, geom_i.hook),
            )
            d_hook_hook = point_point_distance(geom_i.hook, geom_j.hook)
            candidates = [
                ("arm_arm", d_arm_arm, float(thresholds["d_safe_arm_arm_m"])),
                ("arm_hook", d_arm_hook, float(thresholds["d_safe_arm_hook_m"])),
                ("hook_hook", d_hook_hook, float(thresholds["d_safe_hook_hook_m"])),
            ]
            for risk_type, distance, threshold in candidates:
                if distance < threshold and priority[risk_type] > pair_priority:
                    pair_risk = risk_type
                    pair_priority = priority[risk_type]
        if pair_risk is not None:
            low_priority = i if static_by_id[i].priority < static_by_id[j].priority else j
            if priority[pair_risk] > priority.get(risk_by_crane.get(low_priority, ""), 0):
                risk_by_crane[low_priority] = pair_risk
    return risk_by_crane


def short_horizon_risk_pairs(
    statics: list[CraneStatic],
    states: dict[int, CraneState],
    horizon_s: float,
    dt: float,
    thresholds: dict[str, float],
) -> set[int]:
    """Return crane ids that should yield based on constant-velocity short extrapolation."""

    return set(classify_short_horizon_risk(statics, states, horizon_s, dt, thresholds))


def avoidance_command_for_risk(command: Command, risk_type: str, static: CraneStatic) -> Command:
    """Return an axis-aware avoidance command for the dominant risk type."""

    if risk_type == "arm_arm":
        return Command(0.0, command.r_dot_cmd, command.h_dot_cmd, brake_flag=1, emergency_flag=0)
    if risk_type == "arm_hook":
        return Command(0.0, 0.0, command.h_dot_cmd, brake_flag=1, emergency_flag=0)
    if risk_type == "hook_hook":
        upward_h_cmd = min(static.max_h_dot, max(0.0, command.h_dot_cmd, 0.25 * static.max_h_dot))
        return Command(command.theta_dot_cmd, 0.0, upward_h_cmd, brake_flag=1, emergency_flag=0)
    return Command(0.0, 0.0, 0.0, brake_flag=1, emergency_flag=0)


def apply_avoidance(
    commands: dict[int, Command],
    statics: list[CraneStatic],
    states: dict[int, CraneState],
    config: dict[str, Any],
    rng: np.random.Generator,
    dt: float,
    delay_counters: dict[int, int] | None = None,
) -> dict[int, Command]:
    """Modify commands with online short-horizon avoidance rules."""

    interaction_cfg = config["interaction"]
    if not bool(interaction_cfg.get("enabled", True)):
        return commands
    risk_by_crane = classify_short_horizon_risk(
        statics,
        states,
        float(interaction_cfg["short_horizon_check_s"]),
        dt,
        config["risk_thresholds"],
    )
    updated: dict[int, Command] = {}
    fail_prob = float(interaction_cfg.get("avoidance_failure_probability", 0.0))
    error_prob = float(interaction_cfg.get("operator_error_probability", 0.0))
    static_by_id = {static.crane_index: static for static in statics}
    for crane_id, command in commands.items():
        risk_type = risk_by_crane.get(crane_id)
        if risk_type is None:
            if delay_counters is not None:
                delay_counters.pop(crane_id, None)
            updated[crane_id] = command
            continue
        if delay_counters is not None and crane_id not in delay_counters:
            reaction_range = interaction_cfg.get("avoidance_reaction_delay_s_range", [0.0, 0.0])
            communication_range = interaction_cfg.get("communication_delay_s_range", [0.0, 0.0])
            delay_s = float(rng.uniform(float(reaction_range[0]), float(reaction_range[1]))) + float(
                rng.uniform(float(communication_range[0]), float(communication_range[1]))
            )
            delay_counters[crane_id] = int(round(delay_s / max(dt, 1e-6)))
        if delay_counters is not None and delay_counters.get(crane_id, 0) > 0:
            delay_counters[crane_id] -= 1
            updated[crane_id] = command
            continue
        if rng.random() < fail_prob:
            updated[crane_id] = command
            continue
        if rng.random() < error_prob:
            updated[crane_id] = Command(
                -command.theta_dot_cmd,
                command.r_dot_cmd,
                command.h_dot_cmd,
                brake_flag=1,
                emergency_flag=1,
            )
            continue
        if bool(interaction_cfg.get("brake_when_predicted_risk", True)):
            updated[crane_id] = avoidance_command_for_risk(command, risk_type, static_by_id[crane_id])
        else:
            updated[crane_id] = Command(0.0, command.r_dot_cmd, command.h_dot_cmd, brake_flag=1, emergency_flag=0)
    return updated
