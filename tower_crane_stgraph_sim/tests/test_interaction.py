import math

import numpy as np

from tower_sim.dataclasses import Command, CraneState, CraneStatic
from tower_sim.geometry import constant_velocity_extrapolate_clamped, reconstruct_geometry
from tower_sim.interaction import apply_avoidance, avoidance_command_for_risk, compute_pairwise_edge


def make_static(crane_id: int, x: float, y: float, priority: int) -> CraneStatic:
    return CraneStatic(
        scenario_id="scenario_000000",
        scenario_index=0,
        crane_id=f"crane_{crane_id:02d}",
        crane_index=crane_id,
        base_x=x,
        base_y=y,
        base_z=0.0,
        tower_height=30.0,
        jib_length=20.0,
        min_radius=2.0,
        max_radius=20.0,
        safety_radius_arm=2.0,
        safety_radius_hook=2.0,
        max_theta_dot=1.0,
        max_r_dot=1.0,
        max_h_dot=1.0,
        max_theta_acc=1.0,
        max_r_acc=1.0,
        max_h_acc=1.0,
        response_tau=1.0,
        load_capacity=1000.0,
        priority=priority,
    )


def make_state(theta: float, r: float = 10.0, h: float = 20.0) -> CraneState:
    return CraneState(
        theta=theta,
        r=r,
        h=h,
        theta_dot=0.0,
        r_dot=0.0,
        h_dot=0.0,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=0.0,
        task_id=0,
        task_stage="transport_to_dropoff",
    )


def test_pairwise_edge_has_separate_approach_speeds_and_ttc_by_distance_type() -> None:
    static_i = make_static(0, 0.0, 0.0, priority=2)
    static_j = make_static(1, 10.0, -10.0, priority=1)
    state_i = make_state(0.0)
    state_j = make_state(math.pi / 2.0)
    geom_i = reconstruct_geometry(static_i, state_i)
    geom_j = reconstruct_geometry(static_j, state_j)

    edge = compute_pairwise_edge(
        static_i,
        static_j,
        state_i,
        state_j,
        geom_i,
        geom_j,
        dt=1.0,
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    assert "relative_approach_speed_arm_arm" in edge
    assert "relative_approach_speed_arm_hook" in edge
    assert "relative_approach_speed_hook_hook" in edge
    assert edge["relative_approach_speed_arm_arm"] == 0.0
    assert edge["relative_approach_speed_arm_hook"] == 0.0
    assert edge["relative_approach_speed_hook_hook"] == 0.0


def test_pairwise_edge_approach_speed_uses_current_velocity_projection_not_previous_distance() -> None:
    static_i = make_static(0, 0.0, 0.0, priority=2)
    static_j = make_static(1, 25.0, 0.0, priority=1)
    state_i = make_state(0.0, r=10.0)
    state_j = make_state(math.pi, r=10.0)
    state_i.r_dot = 1.0
    state_j.r_dot = 1.0
    geom_i = reconstruct_geometry(static_i, state_i)
    geom_j = reconstruct_geometry(static_j, state_j)

    edge = compute_pairwise_edge(
        static_i,
        static_j,
        state_i,
        state_j,
        geom_i,
        geom_j,
        dt=1.0,
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
    )

    assert edge["relative_approach_speed_hook_hook"] > 0.0


def test_pairwise_edge_approach_speed_ignores_previous_distance_history() -> None:
    static_i = make_static(0, 0.0, 0.0, priority=2)
    static_j = make_static(1, 25.0, 0.0, priority=1)
    state_i = make_state(0.0, r=10.0)
    state_j = make_state(math.pi, r=10.0)
    state_i.r_dot = 1.0
    state_j.r_dot = 1.0
    geom_i = reconstruct_geometry(static_i, state_i)
    geom_j = reconstruct_geometry(static_j, state_j)
    thresholds = {
        "d_safe_arm_arm_m": 1.0,
        "d_safe_arm_hook_m": 1.0,
        "d_safe_hook_hook_m": 1.0,
    }

    first_edge = compute_pairwise_edge(
        static_i,
        static_j,
        state_i,
        state_j,
        geom_i,
        geom_j,
        dt=1.0,
        thresholds=thresholds,
    )
    second_edge = compute_pairwise_edge(
        static_i,
        static_j,
        state_i,
        state_j,
        geom_i,
        geom_j,
        dt=1.0,
        thresholds=thresholds,
    )

    assert first_edge["relative_approach_speed"] == second_edge["relative_approach_speed"]
    assert first_edge["relative_approach_speed_arm_arm"] == second_edge["relative_approach_speed_arm_arm"]
    assert first_edge["relative_approach_speed_arm_hook"] == second_edge["relative_approach_speed_arm_hook"]
    assert first_edge["relative_approach_speed_hook_hook"] == second_edge["relative_approach_speed_hook_hook"]


def test_constant_velocity_extrapolate_clamps_radius_and_height_to_static_limits() -> None:
    static = make_static(0, 0.0, 0.0, priority=1)
    state = make_state(0.0, r=19.5, h=29.5)
    state.r_dot = 5.0
    state.h_dot = 5.0

    extrapolated = constant_velocity_extrapolate_clamped(state, static, horizon_s=1.0, h_clearance=2.0)

    assert extrapolated.r == static.max_radius
    assert extrapolated.h == static.tower_height - 2.0


def test_pairwise_edge_ttc_uses_clamped_boundary_extrapolation() -> None:
    static_i = make_static(0, 0.0, 0.0, priority=2)
    static_j = make_static(1, 44.0, 0.0, priority=1)
    state_i = make_state(0.0, r=20.0)
    state_j = make_state(math.pi, r=20.0)
    state_i.r_dot = 5.0
    state_j.r_dot = 5.0
    geom_i = reconstruct_geometry(static_i, state_i)
    geom_j = reconstruct_geometry(static_j, state_j)

    edge = compute_pairwise_edge(
        static_i,
        static_j,
        state_i,
        state_j,
        geom_i,
        geom_j,
        dt=1.0,
        thresholds={
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
        h_clearance=2.0,
    )

    assert edge["relative_approach_speed_hook_hook"] == 0.0
    assert edge["ttc_est_hook_hook"] == -1.0


def test_arm_arm_avoidance_stops_rotation_first_and_keeps_other_axes() -> None:
    statics = [
        make_static(0, 0.0, 0.0, priority=2),
        make_static(1, 10.0, -10.0, priority=1),
    ]
    states = {
        0: make_state(0.0),
        1: make_state(math.pi / 2.0),
    }
    commands = {
        0: Command(0.5, 0.7, 0.9, brake_flag=0, emergency_flag=0),
        1: Command(0.5, 0.7, 0.9, brake_flag=0, emergency_flag=0),
    }
    config = {
        "risk_thresholds": {
            "d_safe_arm_arm_m": 1.0,
            "d_safe_arm_hook_m": 1.0,
            "d_safe_hook_hook_m": 1.0,
        },
        "interaction": {
            "enabled": True,
            "short_horizon_check_s": 1.0,
            "avoidance_reaction_delay_s_range": [0.0, 0.0],
            "communication_delay_s_range": [0.0, 0.0],
            "avoidance_failure_probability": 0.0,
            "operator_error_probability": 0.0,
            "brake_when_predicted_risk": True,
            "priority_policy": "static_priority",
        },
    }

    updated = apply_avoidance(commands, statics, states, config, np.random.default_rng(0), dt=1.0, delay_counters={})

    assert updated[1].brake_flag == 1
    assert updated[1].theta_dot_cmd == 0.0
    assert updated[1].r_dot_cmd == commands[1].r_dot_cmd
    assert updated[1].h_dot_cmd == commands[1].h_dot_cmd


def test_arm_hook_avoidance_stops_rotation_and_radial_motion() -> None:
    command = Command(0.5, -0.7, 0.9, brake_flag=0, emergency_flag=0)
    static = make_static(1, 10.0, -10.0, priority=1)

    updated = avoidance_command_for_risk(command, "arm_hook", static)

    assert updated.brake_flag == 1
    assert updated.theta_dot_cmd == 0.0
    assert updated.r_dot_cmd == 0.0
    assert updated.h_dot_cmd == command.h_dot_cmd


def test_hook_hook_avoidance_stops_radial_motion_and_raises_hook() -> None:
    command = Command(0.5, -0.7, -0.2, brake_flag=0, emergency_flag=0)
    static = make_static(1, 10.0, -10.0, priority=1)

    updated = avoidance_command_for_risk(command, "hook_hook", static)

    assert updated.brake_flag == 1
    assert updated.theta_dot_cmd == command.theta_dot_cmd
    assert updated.r_dot_cmd == 0.0
    assert updated.h_dot_cmd > 0.0
    assert updated.h_dot_cmd <= static.max_h_dot
