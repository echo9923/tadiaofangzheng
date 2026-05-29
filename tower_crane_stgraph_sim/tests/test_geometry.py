import math

import numpy as np

from tower_sim.geometry import (
    point_point_distance,
    segment_point_distance,
    segment_segment_distance,
    wrap_to_pi,
)


def test_wrap_to_pi_bounds_and_equivalence() -> None:
    assert math.isclose(wrap_to_pi(2.0 * math.pi), 0.0, abs_tol=1e-12)
    assert math.isclose(wrap_to_pi(-2.0 * math.pi), 0.0, abs_tol=1e-12)
    assert -math.pi <= wrap_to_pi(3.5 * math.pi) <= math.pi


def test_segment_segment_distance_intersecting_segments_is_zero() -> None:
    p1 = np.array([0.0, 0.0, 0.0])
    q1 = np.array([2.0, 0.0, 0.0])
    p2 = np.array([1.0, -1.0, 0.0])
    q2 = np.array([1.0, 1.0, 0.0])

    assert segment_segment_distance(p1, q1, p2, q2) == 0.0


def test_segment_segment_distance_parallel_segments() -> None:
    p1 = np.array([0.0, 0.0, 0.0])
    q1 = np.array([2.0, 0.0, 0.0])
    p2 = np.array([0.0, 3.0, 0.0])
    q2 = np.array([2.0, 3.0, 0.0])

    assert math.isclose(segment_segment_distance(p1, q1, p2, q2), 3.0)


def test_segment_segment_distance_closest_point_at_endpoint() -> None:
    p1 = np.array([0.0, 0.0, 0.0])
    q1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([2.0, 1.0, 0.0])
    q2 = np.array([2.0, 2.0, 0.0])

    assert math.isclose(segment_segment_distance(p1, q1, p2, q2), math.sqrt(2.0))


def test_segment_segment_distance_degenerate_segment() -> None:
    p1 = np.array([1.0, 1.0, 1.0])
    q1 = np.array([1.0, 1.0, 1.0])
    p2 = np.array([1.0, 1.0, 3.0])
    q2 = np.array([2.0, 1.0, 3.0])

    assert math.isclose(segment_segment_distance(p1, q1, p2, q2), 2.0)


def test_segment_point_distance_uses_segment_not_infinite_line() -> None:
    p = np.array([0.0, 0.0, 0.0])
    q = np.array([2.0, 0.0, 0.0])
    x = np.array([1.0, 3.0, 0.0])

    assert math.isclose(segment_point_distance(p, q, x), 3.0)


def test_point_point_distance_three_dimensional_height_difference() -> None:
    a = np.array([0.0, 0.0, 5.0])
    b = np.array([3.0, 4.0, 17.0])

    assert math.isclose(point_point_distance(a, b), 13.0)
