from __future__ import annotations

import math
from typing import Any

import numpy as np

from tower_sim.dataclasses import CraneStatic, LiftingTask
from tower_sim.geometry import wrap_to_pi


def _uniform_range(rng: np.random.Generator, value: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(float(value[0]), float(value[1])))


def _angle_toward(src: CraneStatic, dst: CraneStatic) -> float:
    return math.atan2(dst.base_y - src.base_y, dst.base_x - src.base_x)


def _crossing_task(
    scenario_id: int,
    crane: CraneStatic,
    other: CraneStatic,
    task_id: int,
    start_time: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> LiftingTask:
    task_cfg = cfg["task_generation"]
    pickup_h = _uniform_range(rng, task_cfg["pickup_height_range_m"])
    dropoff_h = _uniform_range(rng, task_cfg["dropoff_height_range_m"])
    transport_h = min(
        crane.tower_height - 2.0,
        max(pickup_h, dropoff_h, crane.tower_height * float(task_cfg["transport_height_ratio"])),
    )
    angle_to_other = _angle_toward(crane, other)
    span = rng.uniform(0.35, 0.85)
    pickup_theta = wrap_to_pi(angle_to_other - span)
    dropoff_theta = wrap_to_pi(angle_to_other + span)
    load_ratio = _uniform_range(rng, task_cfg["load_weight_ratio_range"])
    return LiftingTask(
        scenario_id=scenario_id,
        crane_id=crane.crane_id,
        task_id=task_id,
        start_time=start_time,
        pickup_theta=pickup_theta,
        pickup_r=float(rng.uniform(crane.min_radius, crane.max_radius)),
        pickup_h=pickup_h,
        dropoff_theta=dropoff_theta,
        dropoff_r=float(rng.uniform(crane.min_radius, crane.max_radius)),
        dropoff_h=dropoff_h,
        transport_h=transport_h,
        load_weight=load_ratio * crane.load_capacity,
        priority=crane.priority,
    )


def generate_tasks(
    scenario_id: int,
    scene_type: str,
    cranes: list[CraneStatic],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[LiftingTask]:
    """Generate task sequences for all cranes."""

    task_cfg = config["task_generation"]
    min_tasks, max_tasks = int(task_cfg["tasks_per_crane_range"][0]), int(task_cfg["tasks_per_crane_range"][1])
    tasks: list[LiftingTask] = []
    for crane in cranes:
        num_tasks = int(rng.integers(min_tasks, max_tasks + 1))
        for idx in range(num_tasks):
            if scene_type in {"two_crane_crossing", "multi_crane_avoidance", "delayed_or_failed_avoidance"} and len(cranes) > 1:
                other = cranes[(crane.crane_id + 1) % len(cranes)]
                task = _crossing_task(scenario_id, crane, other, idx, start_time=idx * 20.0, cfg=config, rng=rng)
            else:
                pickup_h = _uniform_range(rng, task_cfg["pickup_height_range_m"])
                dropoff_h = _uniform_range(rng, task_cfg["dropoff_height_range_m"])
                transport_h = min(
                    crane.tower_height - 2.0,
                    max(pickup_h, dropoff_h, crane.tower_height * float(task_cfg["transport_height_ratio"])),
                )
                load_ratio = _uniform_range(rng, task_cfg["load_weight_ratio_range"])
                separation = rng.uniform(0.5, 2.4)
                pickup_theta = rng.uniform(-math.pi, math.pi)
                dropoff_theta = wrap_to_pi(pickup_theta + separation * rng.choice([-1.0, 1.0]))
                task = LiftingTask(
                    scenario_id=scenario_id,
                    crane_id=crane.crane_id,
                    task_id=idx,
                    start_time=idx * 20.0,
                    pickup_theta=pickup_theta,
                    pickup_r=float(rng.uniform(crane.min_radius, crane.max_radius)),
                    pickup_h=pickup_h,
                    dropoff_theta=dropoff_theta,
                    dropoff_r=float(rng.uniform(crane.min_radius, crane.max_radius)),
                    dropoff_h=dropoff_h,
                    transport_h=transport_h,
                    load_weight=load_ratio * crane.load_capacity,
                    priority=crane.priority,
                )
            tasks.append(task)
    return tasks
