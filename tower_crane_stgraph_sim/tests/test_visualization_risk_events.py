from pathlib import Path
import uuid

import pandas as pd

from tower_sim.visualization.export import export_risk_explanation_package
from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.plotting import plot_distance_curve
from tower_sim.visualization.risk_events import (
    current_edge_row_for_event,
    edge_series_for_event,
    events_to_frame,
    explain_risk_event,
    extract_risk_events,
)


def test_risk_events_split_one_label_row_into_risk_types() -> None:
    labels = pd.DataFrame(
        [
            {
                "scenario_id": "scenario_000001",
                "scenario_index": 1,
                "timestamp": 3.0,
                "step": 3,
                "horizon_s": 2.0,
                "crane_i": "crane_00",
                "crane_j": "crane_01",
                "crane_i_index": 0,
                "crane_j_index": 1,
                "future_min_d_arm_arm": 0.5,
                "future_min_d_arm_hook_i_to_j": 4.0,
                "future_min_d_arm_hook_j_to_i": 0.7,
                "future_min_d_hook_hook": 8.0,
                "risk_arm_arm": 1,
                "risk_arm_hook_i_to_j": 0,
                "risk_arm_hook_j_to_i": 1,
                "risk_hook_hook": 0,
                "ttc_label_arm_arm": 1.0,
                "ttc_label_arm_hook": 2.0,
                "ttc_label_hook_hook": -1.0,
            }
        ]
    )

    events = extract_risk_events(
        labels,
        config={"risk_thresholds": {"d_safe_arm_arm_m": 1.0, "d_safe_arm_hook_m": 2.0, "d_safe_hook_hook_m": 1.0}},
    )

    assert [event.risk_type for event in events] == ["arm_arm", "arm_hook_j_to_i"]
    assert events[0].future_min_distance == 0.5
    assert events[0].threshold == 1.0
    assert events[1].ttc_label == 2.0
    assert {"risk_type", "future_min_distance", "ttc_label"}.issubset(events_to_frame(events).columns)


def test_risk_event_explanation_and_distance_export() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "test_artifacts" / f"visual_risk_{uuid.uuid4().hex}"
    repo = RunDataRepository(Path(__file__).resolve().parents[1] / "outputs" / "debug_small", prefer_format="csv")
    labels = repo.load_table("edge_future_label").copy()
    labels.loc[0, "risk_arm_arm"] = 1
    labels.loc[0, "future_min_d_arm_arm"] = 0.25
    labels.loc[0, "ttc_label_arm_arm"] = 1.0
    events = extract_risk_events(labels, risk_types=["arm_arm"])
    event = events[0]
    scenario_data = repo.load_scenario_data(event.scenario_id, view_mode="debug")
    current = current_edge_row_for_event(scenario_data.edge_current, event)
    series = edge_series_for_event(scenario_data.edge_current, event)

    text = explain_risk_event(event, current)
    assert "不能作为模型输入特征" in text
    assert "臂-臂" in text
    assert not series.empty

    chart_path = output_dir / "distance.png"
    plot_distance_curve(series, event.risk_type, chart_path, current_timestamp=event.timestamp, horizon_s=event.horizon_s)
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0

    outputs = export_risk_explanation_package(event, scenario_data.edge_current, output_dir / "package")
    assert outputs["chart"].exists()
    assert outputs["markdown"].read_text(encoding="utf-8").startswith("在场景")
    assert outputs["csv"].exists()
