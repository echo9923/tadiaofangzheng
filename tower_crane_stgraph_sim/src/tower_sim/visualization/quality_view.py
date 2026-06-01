from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tower_sim.visualization.i18n_zh import ANOMALY_REASON_LABELS
from tower_sim.visualization.indexing import summarize_run
from tower_sim.visualization.loaders import RunDataRepository


@dataclass(frozen=True)
class QualityViewData:
    report_markdown: str
    risk_ratio_by_scenario: pd.DataFrame
    feature_summary: pd.DataFrame
    plot_paths: list[Path]
    dashboard_metrics: dict[str, Any]
    anomalous_scenarios: pd.DataFrame


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _plot_paths(repository: RunDataRepository) -> list[Path]:
    plots_dir = repository.layout.plots_dir
    if not plots_dir.exists():
        return []
    return sorted(path for path in plots_dir.glob("*.png") if path.is_file())


def build_quality_metrics(repository: RunDataRepository) -> dict[str, Any]:
    metrics = summarize_run(repository)
    quality_path = repository.quality_dir
    risk_ratio = _read_optional_csv(quality_path / "risk_ratio_by_scenario.csv")
    if not risk_ratio.empty and "risk_any_ratio" in risk_ratio:
        metrics["risk_any_ratio_min"] = float(risk_ratio["risk_any_ratio"].min())
        metrics["risk_any_ratio_max"] = float(risk_ratio["risk_any_ratio"].max())
        metrics["risk_any_ratio_mean"] = float(risk_ratio["risk_any_ratio"].mean())
    feature_summary = _read_optional_csv(quality_path / "feature_summary.csv")
    metrics["feature_summary_rows"] = int(len(feature_summary))
    metrics["quality_plots"] = [str(path) for path in _plot_paths(repository)]
    return metrics


def anomalous_scenarios(
    risk_ratio_by_scenario: pd.DataFrame,
    *,
    low: float = 0.02,
    high: float = 0.50,
) -> pd.DataFrame:
    if risk_ratio_by_scenario.empty or "risk_any_ratio" not in risk_ratio_by_scenario.columns:
        return pd.DataFrame(columns=list(risk_ratio_by_scenario.columns) + ["anomaly_reason"])
    result = risk_ratio_by_scenario.copy()
    ratio = pd.to_numeric(result["risk_any_ratio"], errors="coerce")
    reasons = []
    for value in ratio:
        if pd.isna(value):
            reasons.append(ANOMALY_REASON_LABELS["missing risk ratio"])
        elif value < low:
            reasons.append(ANOMALY_REASON_LABELS["risk ratio too low"])
        elif value > high:
            reasons.append(ANOMALY_REASON_LABELS["risk ratio too high"])
        else:
            reasons.append("")
    result["anomaly_reason"] = reasons
    return result[result["anomaly_reason"] != ""].reset_index(drop=True)


def load_quality_view(repository: RunDataRepository, *, low: float = 0.02, high: float = 0.50) -> QualityViewData:
    quality_dir = repository.quality_dir
    report_path = quality_dir / "quality_report.md"
    report_markdown = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    risk_ratio = _read_optional_csv(quality_dir / "risk_ratio_by_scenario.csv")
    feature_summary = _read_optional_csv(quality_dir / "feature_summary.csv")
    return QualityViewData(
        report_markdown=report_markdown,
        risk_ratio_by_scenario=risk_ratio,
        feature_summary=feature_summary,
        plot_paths=_plot_paths(repository),
        dashboard_metrics=build_quality_metrics(repository),
        anomalous_scenarios=anomalous_scenarios(risk_ratio, low=low, high=high),
    )
