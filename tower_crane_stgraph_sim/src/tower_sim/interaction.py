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
    current_min = min(d_arm_arm, d_arm_hook_i_to_j, d_arm_hook_j_to_i, d_hook_hook)
    previous_min = current_min if prev_distances is None else float(prev_distances.get("min", current_min))
    relative_approach_speed = max(0.0, (previous_min - current_min) / max(dt, 1e-6))

    def ttc(distance: float, threshold: float) -> float:
        if distance <= threshold:
            return 0.0
        if relative_approach_speed <= 1e-9:
            return -1.0
        return float((distance - threshold) / relative_approach_speed)

    thresholds = thresholds or {}
    same_height = abs((static_i.base_z + static_i.tower_height) - (static_j.base_z + static_j.tower_height)) < max(
        thresholds.get("d_safe_arm_arm_m", 2.0), 1.0
    )
    base_distance = math.hypot(static_i.base_x - static_j.base_x, static_i.base_y - static_j.base_y)
    return {
        "crane_i": static_i.crane_id,
        "crane_j": static_j.crane_id,
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
        "ttc_est_arm_arm": ttc(d_arm_arm, float(thresholds.get("d_safe_arm_arm_m", static_i.safety_radius_arm))),
        "ttc_est_arm_hook": ttc(
            min(d_arm_hook_i_to_j, d_arm_hook_j_to_i),
            float(thresholds.get("d_safe_arm_hook_m", static_i.safety_radius_hook)),
        ),
        "ttc_est_hook_hook": ttc(d_hook_hook, float(thresholds.get("d_safe_hook_hook_m", static_i.safety_radius_hook))),
        "same_height_risk_zone": int(same_height and overlap_ratio(static_i, static_j) > 0.0),
    }


def compute_edges_for_step(
    statics: list[CraneStatic],
    states: dict[int, CraneState],
    geometries: dict[int, CraneGeometry],
    prev_pair_min: dict[tuple[int, int], float],
    dt: float,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, float | int]], dict[tuple[int, int], float]]:
    """Compute all ordered pair edges for a simulation step."""

    rows: list[dict[str, float | int]] = []
    next_prev: dict[tuple[int, int], float] = {}
    static_by_id = {c.crane_id: c for c in statics}
    for i, j in itertools.permutations(sorted(static_by_id), 2):
        prev = {"min": prev_pair_min[(i, j)]} if (i, j) in prev_pair_min else None
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
        next_prev[(i, j)] = min(
            float(row["d_arm_arm"]),
            float(row["d_arm_hook_i_to_j"]),
            float(row["d_arm_hook_j_to_i"]),
            float(row["d_hook_hook"]),
        )
    return rows, next_prev


def short_horizon_risk_pairs(
    statics: list[CraneStatic],
    states: dict[int, CraneState],
    horizon_s: float,
    dt: float,
    thresholds: dict[str, float],
) -> set[int]:
    """Return crane ids that should yield based on constant-velocity short extrapolation."""

    risky_cranes: set[int] = set()
    static_by_id = {c.crane_id: c for c in statics}
    steps = max(1, int(math.ceil(horizon_s / max(dt, 1e-6))))
    for i, j in itertools.combinations(sorted(static_by_id), 2):
        min_dist = math.inf
        for k in range(1, steps + 1):
            tau = min(horizon_s, k * dt)
            state_i = constant_velocity_extrapolate(states[i], tau)
            state_j = constant_velocity_extrapolate(states[j], tau)
            geom_i = reconstruct_geometry(static_by_id[i], state_i)
            geom_j = reconstruct_geometry(static_by_id[j], state_j)
            d = min(
                segment_segment_distance(geom_i.root, geom_i.tip, geom_j.root, geom_j.tip),
                segment_point_distance(geom_i.root, geom_i.tip, geom_j.hook),
                segment_point_distance(geom_j.root, geom_j.tip, geom_i.hook),
                point_point_distance(geom_i.hook, geom_j.hook),
            )
            min_dist = min(min_dist, d)
        threshold = max(
            float(thresholds["d_safe_arm_arm_m"]),
            float(thresholds["d_safe_arm_hook_m"]),
            float(thresholds["d_safe_hook_hook_m"]),
        )
        if min_dist < threshold:
            low_priority = i if static_by_id[i].priority < static_by_id[j].priority else j
            risky_cranes.add(low_priority)
    return risky_cranes


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
    risky_cranes = short_horizon_risk_pairs(
        statics,
        states,
        float(interaction_cfg["short_horizon_check_s"]),
        dt,
        config["risk_thresholds"],
    )
    updated: dict[int, Command] = {}
    fail_prob = float(interaction_cfg.get("avoidance_failure_probability", 0.0))
    error_prob = float(interaction_cfg.get("operator_error_probability", 0.0))
    for crane_id, command in commands.items():
        if crane_id not in risky_cranes:
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
            updated[crane_id] = Command(0.0, 0.0, 0.0, brake_flag=1, emergency_flag=0)
        else:
            updated[crane_id] = Command(0.0, command.r_dot_cmd, command.h_dot_cmd, brake_flag=1, emergency_flag=0)
    return updated
