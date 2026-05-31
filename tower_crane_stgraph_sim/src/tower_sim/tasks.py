from __future__ import annotations

import math
from typing import Any

import numpy as np

from tower_sim.config import sample_ratio
from tower_sim.dataclasses import CraneStatic, LiftingTask
from tower_sim.geometry import wrap_to_pi
from tower_sim.ids import task_id_from_index


def _uniform_range(rng: np.random.Generator, value: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(float(value[0]), float(value[1])))


def _transport_height(
    crane: CraneStatic,
    pickup_h: float,
    dropoff_h: float,
    task_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    transport_ratio = sample_ratio(rng, task_cfg["transport_height_ratio"])
    return min(
        crane.tower_height - 2.0,
        max(pickup_h, dropoff_h, crane.tower_height * transport_ratio),
    )


def _angle_toward(src: CraneStatic, dst: CraneStatic) -> float:
    return math.atan2(dst.base_y - src.base_y, dst.base_x - src.base_x)


def _crossing_task(
    scenario_index: int,
    crane: CraneStatic,
    other: CraneStatic,
    task_index: int,
    start_time: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> LiftingTask:
    task_cfg = cfg["task_generation"]
    pickup_h = _uniform_range(rng, task_cfg["pickup_height_range_m"])
    dropoff_h = _uniform_range(rng, task_cfg["dropoff_height_range_m"])
    transport_h = _transport_height(crane, pickup_h, dropoff_h, task_cfg, rng)
    angle_to_other = _angle_toward(crane, other)
    span = rng.uniform(0.35, 0.85)
    pickup_theta = wrap_to_pi(angle_to_other - span)
    dropoff_theta = wrap_to_pi(angle_to_other + span)
    load_ratio = _uniform_range(rng, task_cfg["load_weight_ratio_range"])
    return LiftingTask(
        scenario_id=crane.scenario_id,
        scenario_index=scenario_index,
        crane_id=crane.crane_id,
        crane_index=crane.crane_index,
        task_id=task_id_from_index(crane.crane_index, task_index),
        task_index=task_index,
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
    scenario_index: int,
    scene_type: str,
    cranes: list[CraneStatic],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[LiftingTask]:
    """Generate task sequences for all cranes."""

    task_cfg = config["task_generation"]
    min_tasks, max_tasks = int(task_cfg["tasks_per_crane_range"][0]), int(task_cfg["tasks_per_crane_range"][1])
    duration = float(config.get("simulation", {}).get("scenario_duration_s", 0.0))
    latest_start_time = max(0.0, duration * 0.75)
    tasks: list[LiftingTask] = []
    crossing_partners: dict[int, CraneStatic] = {}
    if scene_type == "two_crane_crossing" and len(cranes) >= 2:
        crossing_partners = {
            cranes[0].crane_index: cranes[1],
            cranes[1].crane_index: cranes[0],
        }
    for crane in cranes:
        num_tasks = int(rng.integers(min_tasks, max_tasks + 1))
        start_times = sorted(float(t) for t in rng.uniform(0.0, latest_start_time, size=num_tasks))
        for idx, start_time in enumerate(start_times):
            if scene_type == "two_crane_crossing" and crane.crane_index in crossing_partners:
                other = crossing_partners[crane.crane_index]
                task = _crossing_task(scenario_index, crane, other, idx, start_time=start_time, cfg=config, rng=rng)
            elif scene_type in {"multi_crane_avoidance", "delayed_or_failed_avoidance"} and len(cranes) > 1:
                other = cranes[(cranes.index(crane) + 1) % len(cranes)]
                task = _crossing_task(scenario_index, crane, other, idx, start_time=start_time, cfg=config, rng=rng)
            else:
                pickup_h = _uniform_range(rng, task_cfg["pickup_height_range_m"])
                dropoff_h = _uniform_range(rng, task_cfg["dropoff_height_range_m"])
                transport_h = _transport_height(crane, pickup_h, dropoff_h, task_cfg, rng)
                load_ratio = _uniform_range(rng, task_cfg["load_weight_ratio_range"])
                separation = rng.uniform(0.5, 2.4)
                pickup_theta = rng.uniform(-math.pi, math.pi)
                dropoff_theta = wrap_to_pi(pickup_theta + separation * rng.choice([-1.0, 1.0]))
                task = LiftingTask(
                    scenario_id=crane.scenario_id,
                    scenario_index=scenario_index,
                    crane_id=crane.crane_id,
                    crane_index=crane.crane_index,
                    task_id=task_id_from_index(crane.crane_index, idx),
                    task_index=idx,
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
            tasks.append(task)
    return tasks
