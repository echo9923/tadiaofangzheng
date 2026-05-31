import math

from tower_sim.dataclasses import Command, CraneState, CraneStatic
from tower_sim.dynamics import update_state


def make_static() -> CraneStatic:
    return CraneStatic(
        scenario_id=0,
        crane_id=0,
        base_x=0.0,
        base_y=0.0,
        base_z=0.0,
        tower_height=50.0,
        jib_length=45.0,
        min_radius=5.0,
        max_radius=40.0,
        safety_radius_arm=2.0,
        safety_radius_hook=2.0,
        max_theta_dot=0.4,
        max_r_dot=1.5,
        max_h_dot=1.0,
        max_theta_acc=0.2,
        max_r_acc=0.6,
        max_h_acc=0.5,
        response_tau=1.0,
        load_capacity=10000.0,
        priority=1,
    )


def test_update_state_respects_velocity_and_acceleration_limits() -> None:
    static = make_static()
    state = CraneState(
        theta=0.0,
        r=10.0,
        h=20.0,
        theta_dot=0.0,
        r_dot=0.0,
        h_dot=0.0,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=5000.0,
        task_id=0,
        task_stage="transport_to_dropoff",
    )
    command = Command(
        theta_dot_cmd=10.0,
        r_dot_cmd=10.0,
        h_dot_cmd=10.0,
        brake_flag=0,
        emergency_flag=0,
    )

    updated = update_state(state, static, command, dt=1.0, h_clearance=2.0)
    alpha_load = max(0.5, 1.0 - 0.4 * (state.load_weight / static.load_capacity))

    assert abs(updated.theta_dot) <= static.max_theta_dot
    assert abs(updated.r_dot) <= static.max_r_dot
    assert abs(updated.h_dot) <= static.max_h_dot
    assert abs(updated.theta_ddot) <= static.max_theta_acc * alpha_load + 1e-12
    assert abs(updated.r_ddot) <= static.max_r_acc * alpha_load + 1e-12
    assert abs(updated.h_ddot) <= static.max_h_acc * alpha_load + 1e-12


def test_update_state_clamps_radius_and_height_and_damps_boundary_velocity() -> None:
    static = make_static()
    state = CraneState(
        theta=0.0,
        r=39.9,
        h=47.8,
        theta_dot=0.0,
        r_dot=1.5,
        h_dot=1.0,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=0.0,
        task_id=0,
        task_stage="transport_to_dropoff",
    )
    command = Command(
        theta_dot_cmd=1.5,
        r_dot_cmd=1.5,
        h_dot_cmd=1.0,
        brake_flag=0,
        emergency_flag=0,
    )

    updated = update_state(state, static, command, dt=1.0, h_clearance=2.0)

    assert math.isclose(updated.r, static.max_radius)
    assert math.isclose(updated.h, static.tower_height - 2.0)
    assert abs(updated.r_ddot) <= static.max_r_acc + 1e-12
    assert abs(updated.h_ddot) <= static.max_h_acc + 1e-12


def test_update_state_does_not_create_acceleration_spike_when_clamping_boundaries() -> None:
    static = make_static()
    state = CraneState(
        theta=0.0,
        r=static.max_radius,
        h=static.tower_height - 2.0,
        theta_dot=0.0,
        r_dot=static.max_r_dot,
        h_dot=static.max_h_dot,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=0.0,
        task_id=0,
        task_stage="transport_to_dropoff",
    )

    updated = update_state(
        state,
        static,
        Command(0.0, static.max_r_dot, static.max_h_dot, brake_flag=0, emergency_flag=0),
        dt=1.0,
        h_clearance=2.0,
    )

    assert math.isclose(updated.r, static.max_radius)
    assert math.isclose(updated.h, static.tower_height - 2.0)
    assert abs(updated.r_ddot) <= static.max_r_acc + 1e-12
    assert abs(updated.h_ddot) <= static.max_h_acc + 1e-12


def test_emergency_flag_zeroes_commands_and_uses_stronger_braking() -> None:
    static = make_static()
    state = CraneState(
        theta=0.0,
        r=20.0,
        h=20.0,
        theta_dot=0.4,
        r_dot=1.2,
        h_dot=-0.8,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=0.0,
        task_id=0,
        task_stage="transport_to_dropoff",
    )
    normal_stop = update_state(
        state,
        static,
        Command(0.0, 0.0, 0.0, brake_flag=1, emergency_flag=0),
        dt=1.0,
    )
    emergency_stop = update_state(
        state,
        static,
        Command(10.0, 10.0, 10.0, brake_flag=1, emergency_flag=1),
        dt=1.0,
        emergency_brake_scale=2.0,
    )

    assert abs(emergency_stop.theta_dot) < abs(normal_stop.theta_dot)
    assert abs(emergency_stop.r_dot) < abs(normal_stop.r_dot)
    assert abs(emergency_stop.h_dot) < abs(normal_stop.h_dot)
