from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.risk_events import RiskEvent, extract_risk_events
from tower_sim.visualization.schemas import risk_any_frame


@dataclass(frozen=True)
class ScenarioIndexEntry:
    scenario_id: Any
    scenario_index: int | None
    scene_type: str | None
    split: str | None
    num_cranes: int | None
    risk_any_ratio: float | None = None


def build_scenario_index(repository: RunDataRepository) -> pd.DataFrame:
    """Build the dashboard scenario index from scenario_table plus optional risk ratios."""

    scenarios = repository.load_scenario_table().copy()
    try:
        labels = risk_any_frame(repository.load_table("edge_future_label"))
        if not labels.empty and "scenario_id" in labels.columns:
            ratios = labels.groupby("scenario_id", dropna=False)["risk_any"].mean().rename("risk_any_ratio").reset_index()
            scenarios = scenarios.merge(ratios, on="scenario_id", how="left")
    except FileNotFoundError:
        scenarios["risk_any_ratio"] = pd.NA
    return scenarios


def build_risk_event_index(repository: RunDataRepository, config: dict[str, Any] | None = None) -> list[RiskEvent]:
    return extract_risk_events(repository.load_table("edge_future_label"), config=config)


def scenario_options(repository: RunDataRepository) -> list[Any]:
    scenarios = repository.load_scenario_table()
    if "scenario_id" not in scenarios.columns:
        return []
    return scenarios["scenario_id"].tolist()


def summarize_run(repository: RunDataRepository) -> dict[str, Any]:
    metadata = repository.load_metadata()
    scenarios = repository.load_scenario_table()
    summary: dict[str, Any] = {
        "run_root": str(repository.run_root),
        "num_scenarios": int(scenarios["scenario_id"].nunique()) if "scenario_id" in scenarios else int(len(scenarios)),
        "metadata": metadata,
    }
    if "split" in scenarios:
        summary["split_counts"] = scenarios["split"].value_counts(dropna=False).to_dict()
    if "scene_type" in scenarios:
        summary["scene_type_counts"] = scenarios["scene_type"].value_counts(dropna=False).to_dict()
    if "num_cranes" in scenarios:
        summary["num_cranes_distribution"] = scenarios["num_cranes"].value_counts(dropna=False).sort_index().to_dict()
    try:
        labels = risk_any_frame(repository.load_table("edge_future_label"))
        summary["risk_any_ratio"] = float(labels["risk_any"].mean()) if not labels.empty else 0.0
    except FileNotFoundError:
        summary["risk_any_ratio"] = None
    return summary
