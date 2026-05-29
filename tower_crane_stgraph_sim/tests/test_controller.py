from tower_sim.controller import advance_task_stage, choose_active_task
from tower_sim.dataclasses import CraneState, LiftingTask


def make_task(task_id: int, start_time: float) -> LiftingTask:
    return LiftingTask(
        scenario_id=0,
        crane_id=0,
        task_id=task_id,
        start_time=start_time,
        pickup_theta=0.0,
        pickup_r=10.0,
        pickup_h=5.0,
        dropoff_theta=1.0,
        dropoff_r=12.0,
        dropoff_h=5.0,
        transport_h=20.0,
        load_weight=1000.0,
        priority=1,
    )


def make_state(task_id: int, stage: str) -> CraneState:
    return CraneState(
        theta=1.0,
        r=12.0,
        h=5.0,
        theta_dot=0.0,
        r_dot=0.0,
        h_dot=0.0,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=500.0,
        task_id=task_id,
        task_stage=stage,
    )


def test_release_load_advances_to_next_task_id() -> None:
    state = make_state(0, "release_load")
    task = make_task(0, 0.0)

    updated = advance_task_stage(state, task, stage_tolerance=0.1)

    assert updated.task_stage == "idle_or_next_task"
    assert updated.load_weight == 0.0
    assert updated.task_id == 1


def test_choose_active_task_uses_current_task_id_not_any_started_task() -> None:
    tasks = [make_task(0, 0.0), make_task(1, 10.0), make_task(2, 20.0)]
    state = make_state(1, "idle_or_next_task")

    active = choose_active_task(tasks, state, timestamp=25.0)

    assert active is not None
    assert active.task_id == 1


def test_idle_state_transitions_to_next_task_when_task_is_available() -> None:
    state = make_state(1, "idle_or_next_task")
    task = make_task(1, 10.0)

    updated = advance_task_stage(state, task, stage_tolerance=0.1)

    assert updated.task_id == 1
    assert updated.task_stage == "move_to_pickup"
    assert updated.load_weight == 0.0
