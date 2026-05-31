from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tower_sim.ids import crane_index_from_id, scenario_index_from_id, task_index_from_id


@dataclass(frozen=True)
class CraneStatic:
    """Static parameters of one horizontal trolley tower crane."""

    scenario_id: str
    crane_id: str
    base_x: float
    base_y: float
    base_z: float
    tower_height: float
    jib_length: float
    min_radius: float
    max_radius: float
    safety_radius_arm: float
    safety_radius_hook: float
    max_theta_dot: float
    max_r_dot: float
    max_h_dot: float
    max_theta_acc: float
    max_r_acc: float
    max_h_acc: float
    response_tau: float
    load_capacity: float
    priority: int
    scenario_index: int = 0
    crane_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_index", scenario_index_from_id(self.scenario_id))
        object.__setattr__(self, "crane_index", crane_index_from_id(self.crane_id))


@dataclass
class CraneState:
    """Dynamic state of one crane at a single simulation step."""

    theta: float
    r: float
    h: float
    theta_dot: float
    r_dot: float
    h_dot: float
    theta_ddot: float
    r_ddot: float
    h_ddot: float
    load_weight: float
    task_id: str
    task_stage: str
    task_index: int = 0

    def __post_init__(self) -> None:
        self.task_index = task_index_from_id(self.task_id)


@dataclass
class Command:
    """Velocity command sent to crane drives."""

    theta_dot_cmd: float
    r_dot_cmd: float
    h_dot_cmd: float
    brake_flag: int
    emergency_flag: int


@dataclass(frozen=True)
class LiftingTask:
    """One pickup-transport-dropoff lifting task."""

    scenario_id: str
    crane_id: str
    task_id: str
    start_time: float
    pickup_theta: float
    pickup_r: float
    pickup_h: float
    dropoff_theta: float
    dropoff_r: float
    dropoff_h: float
    transport_h: float
    load_weight: float
    priority: int
    scenario_index: int = 0
    crane_index: int = 0
    task_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_index", scenario_index_from_id(self.scenario_id))
        object.__setattr__(self, "crane_index", crane_index_from_id(self.crane_id))
        object.__setattr__(self, "task_index", task_index_from_id(self.task_id))


@dataclass(frozen=True)
class CraneGeometry:
    """Reconstructed crane geometry at one step."""

    root: np.ndarray
    tip: np.ndarray
    hook: np.ndarray


@dataclass(frozen=True)
class PairwiseEdge:
    """Pairwise physical-prior edge features for a crane pair."""

    crane_i: str
    crane_j: str
    crane_i_index: int
    crane_j_index: int
    d_arm_arm: float
    d_arm_hook_i_to_j: float
    d_arm_hook_j_to_i: float
    d_hook_hook: float
    delta_theta: float
    delta_theta_dot: float
    delta_r: float
    delta_h: float
    delta_tower_height: float
    base_distance: float
    overlap_ratio: float
    relative_approach_speed: float
    relative_approach_speed_arm_arm: float
    relative_approach_speed_arm_hook: float
    relative_approach_speed_hook_hook: float
    ttc_est_arm_arm: float
    ttc_est_arm_hook: float
    ttc_est_hook_hook: float


@dataclass(frozen=True)
class WindowSample:
    """One learning sample for a spatio-temporal graph model."""

    node_features: np.ndarray
    edge_features: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    time_mask: np.ndarray
    y_traj: np.ndarray
    y_risk: np.ndarray
    y_min_distance: np.ndarray
