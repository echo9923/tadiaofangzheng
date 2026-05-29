import pandas as pd

from tower_sim.quality import enforce_quality_gates


def test_quality_gate_raises_on_required_nan_when_enabled() -> None:
    stats = {"has_nan_required": True, "future_leakage": False}
    config = {
        "quality_control": {
            "fail_on_nan_in_required_fields": True,
            "fail_on_future_leakage": True,
        }
    }

    try:
        enforce_quality_gates(stats, config)
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject NaN")


def test_quality_gate_raises_on_future_leakage_when_enabled() -> None:
    stats = {"has_nan_required": False, "future_leakage": True}
    config = {
        "quality_control": {
            "fail_on_nan_in_required_fields": True,
            "fail_on_future_leakage": True,
        }
    }

    try:
        enforce_quality_gates(stats, config)
    except ValueError as exc:
        assert "Future leakage" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject future leakage")
