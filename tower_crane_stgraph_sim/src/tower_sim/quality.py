from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tower_sim.visualization import hist_plot, risk_ratio_plot, sample_scene_topview, sample_time_series
from tower_sim.windowing import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, assert_no_future_leakage, validate_split_disjoint


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def generate_quality_report(
    output_dir: str | Path,
    scenario_table: pd.DataFrame,
    crane_static: pd.DataFrame,
    task_table: pd.DataFrame,
    state_true: pd.DataFrame,
    state_obs: pd.DataFrame,
    edge_current: pd.DataFrame,
    edge_future_label: pd.DataFrame,
    config: dict[str, Any],
    window_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Generate quality report markdown and diagnostic plots."""

    out = Path(output_dir)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hist_plot(state_true, "theta", plots_dir / "theta_distribution.png", "theta distribution")
    hist_plot(state_true, "r", plots_dir / "r_distribution.png", "r distribution")
    hist_plot(state_true, "h", plots_dir / "h_distribution.png", "h distribution")
    hist_plot(edge_current, "d_arm_arm", plots_dir / "d_arm_arm_distribution.png", "arm-arm distance")
    if not edge_current.empty:
        edge_current = edge_current.copy()
        edge_current["d_arm_hook_min"] = edge_current[["d_arm_hook_i_to_j", "d_arm_hook_j_to_i"]].min(axis=1)
    hist_plot(edge_current, "d_arm_hook_min", plots_dir / "d_arm_hook_distribution.png", "arm-hook distance")
    hist_plot(edge_current, "d_hook_hook", plots_dir / "d_hook_hook_distribution.png", "hook-hook distance")
    risk_ratio_plot(edge_future_label, plots_dir / "risk_ratio_distribution.png")
    geometry_path = out / "geometry_table.csv"
    geometry_table = pd.read_csv(geometry_path) if geometry_path.exists() else pd.DataFrame()
    sample_scene_topview(crane_static, geometry_table, plots_dir / "sample_scene_topview.png")
    sample_time_series(state_true, plots_dir / "sample_time_series.png")

    train_ids = scenario_table[scenario_table["split"] == "train"]["scenario_id"].to_numpy()
    val_ids = scenario_table[scenario_table["split"] == "val"]["scenario_id"].to_numpy()
    test_ids = scenario_table[scenario_table["split"] == "test"]["scenario_id"].to_numpy()
    split_ok = True
    try:
        validate_split_disjoint(train_ids, val_ids, test_ids)
    except ValueError:
        split_ok = False

    leakage_ok = True
    try:
        assert_no_future_leakage(NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES)
    except ValueError:
        leakage_ok = False

    r_out = False
    h_out = False
    speed_out = False
    acc_out = False
    if not state_true.empty and not crane_static.empty:
        merged = state_true.merge(crane_static, on=["scenario_id", "crane_id"], how="left")
        r_out = bool(((merged["r"] < merged["min_radius"] - 1e-6) | (merged["r"] > merged["max_radius"] + 1e-6)).any())
        h_out = bool(((merged["h"] < -1e-6) | (merged["h"] > merged["tower_height"] - 2.0 + 1e-6)).any())
        speed_out = bool(
            (
                (merged["theta_dot"].abs() > merged["max_theta_dot"] + 1e-6)
                | (merged["r_dot"].abs() > merged["max_r_dot"] + 1e-6)
                | (merged["h_dot"].abs() > merged["max_h_dot"] + 1e-6)
            ).any()
        )
        acc_out = bool(
            (
                (merged["theta_ddot"].abs() > merged["max_theta_acc"] + 1e-6)
                | (merged["r_ddot"].abs() > merged["max_r_acc"] + 1e-6)
                | (merged["h_ddot"].abs() > merged["max_h_acc"] + 1e-6)
            ).any()
        )

    labels = edge_future_label
    if labels.empty:
        risk_arm_arm = risk_arm_hook = risk_hook_hook = risk_any = 0.0
    else:
        risk_arm_arm = _ratio(labels["risk_arm_arm"])
        risk_arm_hook = _ratio(labels[["risk_arm_hook_i_to_j", "risk_arm_hook_j_to_i"]].max(axis=1))
        risk_hook_hook = _ratio(labels["risk_hook_hook"])
        risk_any = _ratio(labels[["risk_arm_arm", "risk_arm_hook_i_to_j", "risk_arm_hook_j_to_i", "risk_hook_hook"]].max(axis=1))

    stats = {
        "num_scenarios": int(scenario_table["scenario_id"].nunique()) if not scenario_table.empty else 0,
        "num_tasks": int(len(task_table)),
        "total_duration_s": float(scenario_table["duration_s"].sum()) if not scenario_table.empty else 0.0,
        "sampling_frequency_hz": 1.0 / float(config["simulation"]["dt"]),
        "risk_arm_arm_ratio": risk_arm_arm,
        "risk_arm_hook_ratio": risk_arm_hook,
        "risk_hook_hook_ratio": risk_hook_hook,
        "risk_any_ratio": risk_any,
        "has_nan_required": bool(
            state_true.isna().any().any()
            or crane_static.isna().any().any()
            or edge_current.isna().any().any()
            or edge_future_label.isna().any().any()
        ),
        "r_out_of_bounds": r_out,
        "h_out_of_bounds": h_out,
        "speed_out_of_bounds": speed_out,
        "acc_out_of_bounds": acc_out,
        "split_disjoint": split_ok,
        "future_leakage": not leakage_ok,
    }

    crane_dist = scenario_table["num_cranes"].value_counts().sort_index().to_dict() if not scenario_table.empty else {}
    lines = [
        "# Quality Report",
        "",
        f"- Scenario count: {stats['num_scenarios']}",
        f"- Crane count distribution: {crane_dist}",
        f"- Total simulation duration: {stats['total_duration_s']:.3f} s",
        f"- Sampling frequency: {stats['sampling_frequency_hz']:.3f} Hz",
        f"- Task count: {stats['num_tasks']}",
        f"- Arm-arm risk positive ratio: {risk_arm_arm:.6f}",
        f"- Arm-hook risk positive ratio: {risk_arm_hook:.6f}",
        f"- Hook-hook risk positive ratio: {risk_hook_hook:.6f}",
        f"- Overall risk positive ratio: {risk_any:.6f}",
        "",
        "## State Distribution",
        "",
        state_true[["theta", "r", "h", "theta_dot", "r_dot", "h_dot", "theta_ddot", "r_ddot", "h_ddot"]]
        .describe()
        .to_markdown(),
        "",
        "## Distance Distribution",
        "",
        edge_current[["d_arm_arm", "d_arm_hook_i_to_j", "d_arm_hook_j_to_i", "d_hook_hook"]].describe().to_markdown()
        if not edge_current.empty
        else "No edge records.",
        "",
        "## Integrity Checks",
        "",
        f"- NaN in required fields: {_yes_no(stats['has_nan_required'])}",
        f"- r out of bounds: {_yes_no(r_out)}",
        f"- h out of bounds: {_yes_no(h_out)}",
        f"- velocity out of bounds: {_yes_no(speed_out)}",
        f"- acceleration out of bounds: {_yes_no(acc_out)}",
        f"- train/val/test scenario_id disjoint: {_yes_no(split_ok)}",
        f"- future label leakage detected: {_yes_no(not leakage_ok)}",
        f"- random seed: {config['project']['random_seed']}",
        f"- config_used.yaml: {out / 'config_used.yaml'}",
        "",
        "## Window Counts",
        "",
        str(window_counts or {}),
        "",
        "## Plots",
        "",
        "- plots/theta_distribution.png",
        "- plots/r_distribution.png",
        "- plots/h_distribution.png",
        "- plots/d_arm_arm_distribution.png",
        "- plots/d_arm_hook_distribution.png",
        "- plots/d_hook_hook_distribution.png",
        "- plots/risk_ratio_distribution.png",
        "- plots/sample_scene_topview.png",
        "- plots/sample_time_series.png",
        "",
        "Safety distance thresholds are simulation parameters for controlled experiments, not normative construction-code values.",
    ]
    (out / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    return stats
