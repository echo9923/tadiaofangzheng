from __future__ import annotations

import math
from typing import Any

import numpy as np

from tower_sim.dataclasses import CraneStatic

SCENE_TYPES = (
    "no_overlap_safe",
    "overlap_no_conflict",
    "two_crane_crossing",
    "multi_crane_avoidance",
    "delayed_or_failed_avoidance",
)


def sample_scene_type(rng: np.random.Generator, ratios: dict[str, float]) -> str:
    """Sample one scene type from configured ratios."""

    labels = [label for label in SCENE_TYPES if float(ratios.get(label, 0.0)) > 0.0]
    if not labels:
        labels = list(SCENE_TYPES)
    weights = np.array([float(ratios.get(label, 0.0)) for label in labels], dtype=float)
    if weights.sum() <= 0.0:
        weights[:] = 1.0
    weights /= weights.sum()
    return str(rng.choice(labels, p=weights))


def _uniform_range(rng: np.random.Generator, value: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(float(value[0]), float(value[1])))


def _sample_base_positions(
    rng: np.random.Generator,
    scene_type: str,
    num_cranes: int,
    site_size_m: list[float],
    min_base_distance_m: float,
    typical_jib: float,
) -> list[tuple[float, float]]:
    length, width = float(site_size_m[0]), float(site_size_m[1])
    margin = max(10.0, min_base_distance_m)
    center = np.array([length * 0.5, width * 0.5], dtype=float)
    bases: list[tuple[float, float]] = []

    if scene_type == "no_overlap_safe":
        for i in range(num_cranes):
            for _ in range(1000):
                candidate = np.array(
                    [
                        rng.uniform(margin, max(margin + 1.0, length - margin)),
                        rng.uniform(margin, max(margin + 1.0, width - margin)),
                    ],
                    dtype=float,
                )
                if all(np.linalg.norm(candidate - np.array(p)) >= typical_jib * 1.6 for p in bases):
                    bases.append((float(candidate[0]), float(candidate[1])))
                    break
            else:
                angle = 2.0 * math.pi * i / max(num_cranes, 1)
                point = center + typical_jib * 1.8 * np.array([math.cos(angle), math.sin(angle)])
                bases.append((float(np.clip(point[0], margin, length - margin)), float(np.clip(point[1], margin, width - margin))))
        return bases

    radius = typical_jib * (0.85 if scene_type in {"two_crane_crossing", "delayed_or_failed_avoidance"} else 1.05)
    for i in range(num_cranes):
        angle = 2.0 * math.pi * i / max(num_cranes, 1) + rng.uniform(-0.25, 0.25)
        jitter = rng.normal(0.0, typical_jib * 0.08, size=2)
        point = center + radius * np.array([math.cos(angle), math.sin(angle)]) + jitter
        bases.append((float(np.clip(point[0], margin, length - margin)), float(np.clip(point[1], margin, width - margin))))

    if scene_type == "two_crane_crossing" and num_cranes >= 2:
        bases[0] = (float(center[0] - typical_jib * 0.35), float(center[1]))
        bases[1] = (float(center[0] + typical_jib * 0.35), float(center[1]))
    return bases


def generate_cranes(
    scenario_id: int,
    scene_type: str,
    num_cranes: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[CraneStatic]:
    """Generate crane static parameters and layout for one scenario."""

    crane_cfg = config["crane_static"]
    motion_cfg = config["motion_limits"]
    layout_cfg = config["layout"]
    jib_mid = float(np.mean(crane_cfg["jib_length_range_m"]))
    bases = _sample_base_positions(
        rng,
        scene_type,
        num_cranes,
        layout_cfg["site_size_m"],
        float(layout_cfg["min_base_distance_m"]),
        jib_mid,
    )
    cranes: list[CraneStatic] = []
    clustered_tower_height = _uniform_range(rng, crane_cfg["tower_height_range_m"])
    for crane_id, (base_x, base_y) in enumerate(bases):
        if scene_type in {"two_crane_crossing", "multi_crane_avoidance", "delayed_or_failed_avoidance"}:
            tower_height = float(
                np.clip(
                    clustered_tower_height + rng.normal(0.0, 0.7),
                    float(crane_cfg["tower_height_range_m"][0]),
                    float(crane_cfg["tower_height_range_m"][1]),
                )
            )
        else:
            tower_height = _uniform_range(rng, crane_cfg["tower_height_range_m"])
        jib_length = _uniform_range(rng, crane_cfg["jib_length_range_m"])
        min_radius = _uniform_range(rng, crane_cfg["min_radius_range_m"])
        margin = float(crane_cfg["max_radius_margin_m"])
        max_radius = max(min_radius + 1.0, jib_length - margin)
        cranes.append(
            CraneStatic(
                scenario_id=scenario_id,
                crane_id=crane_id,
                base_x=base_x,
                base_y=base_y,
                base_z=0.0,
                tower_height=tower_height,
                jib_length=jib_length,
                min_radius=min_radius,
                max_radius=max_radius,
                safety_radius_arm=float(crane_cfg["safety_radius_arm_m"]),
                safety_radius_hook=float(crane_cfg["safety_radius_hook_m"]),
                max_theta_dot=_uniform_range(rng, motion_cfg["max_theta_dot_range_rad_s"]),
                max_r_dot=_uniform_range(rng, motion_cfg["max_r_dot_range_m_s"]),
                max_h_dot=_uniform_range(rng, motion_cfg["max_h_dot_range_m_s"]),
                max_theta_acc=_uniform_range(rng, motion_cfg["max_theta_acc_range_rad_s2"]),
                max_r_acc=_uniform_range(rng, motion_cfg["max_r_acc_range_m_s2"]),
                max_h_acc=_uniform_range(rng, motion_cfg["max_h_acc_range_m_s2"]),
                response_tau=_uniform_range(rng, motion_cfg["response_tau_s_range"]),
                load_capacity=_uniform_range(rng, crane_cfg["load_capacity_range_kg"]),
                priority=int(num_cranes - crane_id),
            )
        )
    return cranes


def overlap_ratio(a: CraneStatic, b: CraneStatic) -> float:
    """Estimate plan-view operating radius overlap between two cranes."""

    d = math.hypot(a.base_x - b.base_x, a.base_y - b.base_y)
    radius_sum = a.max_radius + b.max_radius
    if radius_sum <= 0.0 or d >= radius_sum:
        return 0.0
    return float(np.clip((radius_sum - d) / max(min(a.max_radius, b.max_radius), 1e-6), 0.0, 1.0))
