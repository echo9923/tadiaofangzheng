from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from tower_sim.dataclasses import CraneGeometry, CraneState, CraneStatic

EPS = 1e-12


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi]."""

    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=EPS) and angle > 0:
        return math.pi
    if math.isclose(wrapped, math.pi, abs_tol=EPS) and angle < 0:
        return -math.pi
    return wrapped


def point_point_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two 3D points."""

    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def segment_point_distance(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    """Return the shortest distance between segment pq and point x."""

    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    v = q_arr - p_arr
    denom = float(np.dot(v, v))
    if denom <= EPS:
        return point_point_distance(p_arr, x_arr)
    t = float(np.dot(x_arr - p_arr, v) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    closest = p_arr + t * v
    return point_point_distance(closest, x_arr)


def segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Return the shortest distance between two finite 3D line segments.

    The implementation follows the standard closest-points formulation and
    explicitly handles degenerate point-like segments.
    """

    p1_arr = np.asarray(p1, dtype=float)
    q1_arr = np.asarray(q1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    q2_arr = np.asarray(q2, dtype=float)

    d1 = q1_arr - p1_arr
    d2 = q2_arr - p2_arr
    r = p1_arr - p2_arr
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= EPS and e <= EPS:
        return point_point_distance(p1_arr, p2_arr)
    if a <= EPS:
        return segment_point_distance(p2_arr, q2_arr, p1_arr)
    if e <= EPS:
        return segment_point_distance(p1_arr, q1_arr, p2_arr)

    b = float(np.dot(d1, d2))
    c = float(np.dot(d1, r))
    denom = a * e - b * b
    if denom != 0.0:
        s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
    else:
        s = 0.0

    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = float(np.clip(-c / a, 0.0, 1.0))
    elif t > 1.0:
        t = 1.0
        s = float(np.clip((b - c) / a, 0.0, 1.0))

    closest_1 = p1_arr + s * d1
    closest_2 = p2_arr + t * d2
    dist = point_point_distance(closest_1, closest_2)
    if dist < 1e-10:
        return 0.0
    return dist


def reconstruct_geometry(static: CraneStatic, state: CraneState) -> CraneGeometry:
    """Reconstruct jib root, jib tip, and hook point from state variables."""

    root = np.array(
        [static.base_x, static.base_y, static.base_z + static.tower_height],
        dtype=float,
    )
    direction = np.array([math.cos(state.theta), math.sin(state.theta), 0.0], dtype=float)
    tip = root + static.jib_length * direction
    hook = np.array(
        [
            static.base_x + state.r * math.cos(state.theta),
            static.base_y + state.r * math.sin(state.theta),
            state.h,
        ],
        dtype=float,
    )
    return CraneGeometry(root=root, tip=tip, hook=hook)


def reconstruct_geometry_from_rows(static_row: Mapping[str, float], state_row: Mapping[str, float]) -> CraneGeometry:
    """Reconstruct geometry from pandas/dict rows without allocating dataclasses."""

    root = np.array(
        [
            float(static_row["base_x"]),
            float(static_row["base_y"]),
            float(static_row["base_z"]) + float(static_row["tower_height"]),
        ],
        dtype=float,
    )
    theta = float(state_row["theta"])
    direction = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
    tip = root + float(static_row["jib_length"]) * direction
    hook = np.array(
        [
            float(static_row["base_x"]) + float(state_row["r"]) * math.cos(theta),
            float(static_row["base_y"]) + float(state_row["r"]) * math.sin(theta),
            float(state_row["h"]),
        ],
        dtype=float,
    )
    return CraneGeometry(root=root, tip=tip, hook=hook)


def constant_velocity_extrapolate(state: CraneState, horizon_s: float) -> CraneState:
    """Return a constant-velocity extrapolation used for online short checks."""

    return CraneState(
        theta=wrap_to_pi(state.theta + state.theta_dot * horizon_s),
        r=state.r + state.r_dot * horizon_s,
        h=state.h + state.h_dot * horizon_s,
        theta_dot=state.theta_dot,
        r_dot=state.r_dot,
        h_dot=state.h_dot,
        theta_ddot=0.0,
        r_ddot=0.0,
        h_ddot=0.0,
        load_weight=state.load_weight,
        task_id=state.task_id,
        task_index=state.task_index,
        task_stage=state.task_stage,
    )
