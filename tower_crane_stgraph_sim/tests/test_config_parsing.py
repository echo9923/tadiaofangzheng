import numpy as np

from tower_sim.config import apply_overrides, get_command_smoothing, get_stage_tolerance, sample_ratio


def test_stage_tolerance_accepts_component_mapping() -> None:
    cfg = {"stage_tolerance": {"theta_rad": 0.03, "r_m": 0.4, "h_m": 0.5}}

    assert get_stage_tolerance(cfg) == (0.03, 0.4, 0.5)


def test_stage_tolerance_keeps_scalar_configs_backward_compatible() -> None:
    cfg = {"stage_tolerance": 1.2}

    assert get_stage_tolerance(cfg) == (1.2, 1.2, 1.2)


def test_sample_ratio_accepts_scalar_or_range() -> None:
    rng = np.random.default_rng(123)

    assert sample_ratio(rng, 0.7) == 0.7
    sampled = sample_ratio(rng, [0.55, 0.85])

    assert 0.55 <= sampled <= 0.85


def test_boolean_command_smoothing_uses_safe_default_not_one() -> None:
    assert get_command_smoothing({"command_smoothing": True}) == 0.2
    assert get_command_smoothing({"command_smoothing": False}) == 0.0


def test_apply_overrides_supports_formal_cli_options() -> None:
    base = {
        "project": {"random_seed": 1, "output_dir": "outputs/base"},
        "simulation": {"num_scenarios": 2, "scenario_duration_s": 120.0, "dt": 1.0},
    }

    cfg = apply_overrides(
        base,
        num_scenarios=20,
        output_dir="outputs/debug_run",
        duration=600.0,
        seed=42,
        dt=0.2,
        run_id="run_001",
    )

    assert cfg["simulation"]["num_scenarios"] == 20
    assert cfg["simulation"]["scenario_duration_s"] == 600.0
    assert cfg["simulation"]["dt"] == 0.2
    assert cfg["project"]["random_seed"] == 42
    assert cfg["project"]["output_dir"] == "outputs/debug_run"
    assert cfg["project"]["run_id"] == "run_001"
