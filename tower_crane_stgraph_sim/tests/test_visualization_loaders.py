from pathlib import Path
import uuid

import pandas as pd

from tower_sim.visualization.animation_player import get_animation_frame
from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.schemas import ViewMode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_repository_loads_legacy_debug_small_layout() -> None:
    repo = RunDataRepository(_repo_root() / "outputs" / "debug_small", prefer_format="csv")

    assert repo.tables_dir == _repo_root() / "outputs" / "debug_small"
    assert repo.windows_dir == _repo_root() / "outputs" / "debug_small"
    assert repo.quality_dir == _repo_root() / "outputs" / "debug_small"
    assert repo.validate(required_tables=["scenario_table", "crane_static", "state_obs", "geometry_table", "edge_current"]) == []

    scenarios = repo.list_scenarios()
    assert not scenarios.empty
    assert {"scenario_id", "split", "num_cranes"}.issubset(scenarios.columns)


def test_repository_loads_formal_tables_directory_and_prefers_csv_when_configured() -> None:
    run_root = _repo_root() / "test_artifacts" / f"visual_formal_{uuid.uuid4().hex}" / "run_001"
    tables = run_root / "tables"
    tables.mkdir(parents=True)
    (run_root / "config_used.yaml").write_text("simulation:\n  dt: 1.0\n", encoding="utf-8")
    for table_name in [
        "scenario_table",
        "crane_static",
        "task_table",
        "state_true",
        "state_obs",
        "geometry_table",
        "edge_current",
        "edge_future_label",
    ]:
        pd.DataFrame([{"scenario_id": "scenario_000000", "step": 0}]).to_csv(tables / f"{table_name}.csv", index=False)

    repo = RunDataRepository(run_root, prefer_format="csv")

    assert repo.tables_dir == tables
    assert repo.validate() == []
    assert repo.load_config()["simulation"]["dt"] == 1.0
    assert repo.table_path("scenario_table") == tables / "scenario_table.csv"


def test_input_view_frame_does_not_load_future_labels() -> None:
    repo = RunDataRepository(_repo_root() / "outputs" / "debug_small", prefer_format="csv")
    frame = get_animation_frame(repo, 0, step=0, view_mode=ViewMode.INPUT)

    assert frame.view_mode is ViewMode.INPUT
    assert frame.labels is None
    assert not any("future_min_d" in column for column in frame.edges.columns)
    assert not any(column.startswith("risk_") for column in frame.edges.columns)


def test_debug_view_frame_includes_label_rows_when_available() -> None:
    repo = RunDataRepository(_repo_root() / "outputs" / "debug_small", prefer_format="csv")
    frame = get_animation_frame(repo, 0, step=0, view_mode=ViewMode.DEBUG)

    assert frame.labels is not None
    assert not frame.labels.empty
    assert {"future_min_d_arm_arm", "risk_arm_arm"}.issubset(frame.labels.columns)
