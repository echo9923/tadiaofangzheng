from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tower_sim.visualization import hist_plot, risk_ratio_plot, sample_scene_topview, sample_time_series
from tower_sim.windowing import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, assert_no_future_leakage, validate_named_splits_disjoint


def enforce_quality_gates(stats: dict[str, Any], config: dict[str, Any]) -> None:
    """Raise when configured quality gates fail."""

    qc = config.get("quality_control", {})
    if bool(qc.get("fail_on_nan_in_required_fields", False)) and bool(stats.get("has_nan_required", False)):
        raise ValueError("NaN found in required fields")
    if bool(qc.get("fail_on_future_leakage", False)) and bool(stats.get("future_leakage", False)):
        raise ValueError("Future leakage detected")
    fail_on_risk_ratio = bool(qc.get("fail_on_risk_ratio_out_of_range", "target_risk_ratio_range" in qc))
    if fail_on_risk_ratio and "target_risk_ratio_range" in qc and not bool(stats.get("risk_ratio_in_target_range", True)):
        low, high = [float(x) for x in qc["target_risk_ratio_range"]]
        raise ValueError(f"risk_any_ratio={float(stats.get('risk_any_ratio', 0.0)):.4f} outside [{low}, {high}]")
    gate_failures = [
        "r_out_of_bounds",
        "h_out_of_bounds",
        "speed_out_of_bounds",
        "acc_out_of_bounds",
        "edge_future_label_has_inf",
        "no_overlap_safe_has_overlap",
    ]
    for key in gate_failures:
        if bool(stats.get(key, False)):
            raise ValueError(f"Quality gate failed: {key}")
    if not bool(stats.get("split_disjoint", True)):
        raise ValueError("Quality gate failed: split_disjoint")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def _has_inf(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    return bool(np.isinf(numeric.to_numpy()).any())


def _risk_ratio_by_scenario(edge_future_label: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario_id",
        "scenario_uid",
        "risk_arm_arm_ratio",
        "risk_arm_hook_ratio",
        "risk_hook_hook_ratio",
        "risk_any_ratio",
    ]
    if edge_future_label.empty:
        return pd.DataFrame(columns=columns)
    labels = edge_future_label.copy()
    labels["risk_arm_hook_any"] = labels[["risk_arm_hook_i_to_j", "risk_arm_hook_j_to_i"]].max(axis=1)
    labels["risk_any"] = labels[["risk_arm_arm", "risk_arm_hook_i_to_j", "risk_arm_hook_j_to_i", "risk_hook_hook"]].max(axis=1)
    group_keys = ["scenario_id"]
    if "scenario_uid" in labels.columns:
        group_keys.append("scenario_uid")
    summary = (
        labels.groupby(group_keys, dropna=False)
        .agg(
            risk_arm_arm_ratio=("risk_arm_arm", "mean"),
            risk_arm_hook_ratio=("risk_arm_hook_any", "mean"),
            risk_hook_hook_ratio=("risk_hook_hook", "mean"),
            risk_any_ratio=("risk_any", "mean"),
        )
        .reset_index()
    )
    if "scenario_uid" not in summary.columns:
        summary["scenario_uid"] = summary["scenario_id"].astype(str)
    return summary[columns]


def _feature_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, df in tables.items():
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        summary = numeric.describe().transpose().reset_index().rename(columns={"index": "feature"})
        summary.insert(0, "source", source)
        frames.append(summary)
    if not frames:
        return pd.DataFrame(columns=["source", "feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    return pd.concat(frames, ignore_index=True)


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
    geometry_table: pd.DataFrame | None = None,
    config_used_path: str | Path | None = None,
    window_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Generate quality report markdown and diagnostic plots."""

    out = Path(output_dir)
    generate_plots = bool(config.get("quality_control", {}).get("generate_summary_plots", True))
    plot_lines: list[str]
    if generate_plots:
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
        if geometry_table is None:
            geometry_path = out / "geometry_table.csv"
            geometry_table = pd.read_csv(geometry_path) if geometry_path.exists() else pd.DataFrame()
        sample_scene_topview(crane_static, geometry_table, plots_dir / "sample_scene_topview.png")
        sample_time_series(state_true, plots_dir / "sample_time_series.png")
        plot_lines = [
            "- plots/theta_distribution.png",
            "- plots/r_distribution.png",
            "- plots/h_distribution.png",
            "- plots/d_arm_arm_distribution.png",
            "- plots/d_arm_hook_distribution.png",
            "- plots/d_hook_hook_distribution.png",
            "- plots/risk_ratio_distribution.png",
            "- plots/sample_scene_topview.png",
            "- plots/sample_time_series.png",
        ]
    else:
        if not edge_current.empty:
            edge_current = edge_current.copy()
            edge_current["d_arm_hook_min"] = edge_current[["d_arm_hook_i_to_j", "d_arm_hook_j_to_i"]].min(axis=1)
        if geometry_table is None:
            geometry_table = pd.DataFrame()
        plot_lines = ["Plots disabled by quality_control.generate_summary_plots=false."]

    split_key = "scenario_index" if "scenario_index" in scenario_table.columns else "scenario_id"
    split_ids = {
        str(split): scenario_table[scenario_table["split"] == split][split_key].to_numpy()
        for split in scenario_table["split"].dropna().unique()
    } if "split" in scenario_table.columns else {}
    split_ok = True
    try:
        validate_named_splits_disjoint(split_ids)
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
        merge_keys = ["scenario_index", "crane_index"] if {"scenario_index", "crane_index"}.issubset(state_true.columns) else ["scenario_id", "crane_id"]
        merged = state_true.merge(crane_static, on=merge_keys, how="left", suffixes=("", "_static"))
        r_out = bool(((merged["r"] < merged["min_radius"] - 1e-6) | (merged["r"] > merged["max_radius"] + 1e-6)).any())
        h_clearance = float(config.get("dynamics", {}).get("hook_clearance_m", 2.0))
        h_out = bool(((merged["h"] < -1e-6) | (merged["h"] > merged["tower_height"] - h_clearance + 1e-6)).any())
        speed_out = bool(
            (
                (merged["theta_dot"].abs() > merged["max_theta_dot"] + 1e-6)
                | (merged["r_dot"].abs() > merged["max_r_dot"] + 1e-6)
                | (merged["h_dot"].abs() > merged["max_h_dot"] + 1e-6)
            ).any()
        )
        emergency_scale = max(1.0, float(config.get("dynamics", {}).get("emergency_brake_scale", 2.0)))
        normal_scale = max(1.0, float(config.get("dynamics", {}).get("normal_brake_scale", 1.0)))
        emergency_flag = pd.to_numeric(merged["emergency_flag"], errors="coerce").fillna(0)
        brake_flag = pd.to_numeric(merged["brake_flag"], errors="coerce").fillna(0)
        acc_scale = np.where(emergency_flag > 0, emergency_scale, np.where(brake_flag > 0, normal_scale, 1.0))
        acc_out = bool(
            (
                (merged["theta_ddot"].abs() > merged["max_theta_acc"] * acc_scale + 1e-6)
                | (merged["r_ddot"].abs() > merged["max_r_acc"] * acc_scale + 1e-6)
                | (merged["h_ddot"].abs() > merged["max_h_acc"] * acc_scale + 1e-6)
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
    low, high = [float(x) for x in config.get("quality_control", {}).get("target_risk_ratio_range", [0.0, 1.0])]
    risk_ratio_in_target_range = bool(low <= risk_any <= high)
    edge_future_label_has_inf = _has_inf(edge_future_label)

    no_overlap_safe_has_overlap = False
    if not scenario_table.empty and not crane_static.empty and "scene_type" in scenario_table.columns:
        for scenario_id in scenario_table.loc[scenario_table["scene_type"] == "no_overlap_safe", "scenario_id"].unique():
            statics = crane_static[crane_static["scenario_id"] == scenario_id]
            static_rows = [row for _, row in statics.iterrows()]
            for idx, row_i in enumerate(static_rows):
                for row_j in static_rows[idx + 1 :]:
                    base_distance = float(np.hypot(float(row_i["base_x"]) - float(row_j["base_x"]), float(row_i["base_y"]) - float(row_j["base_y"])))
                    if base_distance < float(row_i["max_radius"]) + float(row_j["max_radius"]) - 1e-6:
                        no_overlap_safe_has_overlap = True
                        break
                if no_overlap_safe_has_overlap:
                    break

    stats = {
        "num_scenarios": int(scenario_table["scenario_id"].nunique()) if not scenario_table.empty else 0,
        "num_tasks": int(len(task_table)),
        "total_duration_s": float(scenario_table["duration_s"].sum()) if not scenario_table.empty else 0.0,
        "sampling_frequency_hz": 1.0 / float(config["simulation"]["dt"]),
        "risk_arm_arm_ratio": risk_arm_arm,
        "risk_arm_hook_ratio": risk_arm_hook,
        "risk_hook_hook_ratio": risk_hook_hook,
        "risk_any_ratio": risk_any,
        "risk_ratio_in_target_range": risk_ratio_in_target_range,
        "edge_future_label_has_inf": edge_future_label_has_inf,
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
        "no_overlap_safe_has_overlap": no_overlap_safe_has_overlap,
    }

    risk_ratio_by_scenario = _risk_ratio_by_scenario(edge_future_label)
    risk_ratio_by_scenario.to_csv(out / "risk_ratio_by_scenario.csv", index=False, encoding="utf-8")
    feature_summary = _feature_summary(
        {
            "state_true": state_true,
            "edge_current": edge_current,
            "edge_future_label": edge_future_label,
        }
    )
    feature_summary.to_csv(out / "feature_summary.csv", index=False, encoding="utf-8")

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
        f"- Risk ratio target range: [{low:.6f}, {high:.6f}]",
        f"- Risk ratio in target range: {_yes_no(risk_ratio_in_target_range)}",
        "",
        "## State Distribution",
        "",
        state_true[["theta", "r", "h", "theta_dot", "r_dot", "h_dot", "theta_ddot", "r_ddot", "h_ddot"]]
        .describe()
        .to_markdown()
        if {"theta", "r", "h", "theta_dot", "r_dot", "h_dot", "theta_ddot", "r_ddot", "h_ddot"}.issubset(state_true.columns)
        else "No state records.",
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
        f"- all scenario splits disjoint: {_yes_no(split_ok)}",
        f"- future label leakage detected: {_yes_no(not leakage_ok)}",
        f"- edge_future_label has inf: {_yes_no(edge_future_label_has_inf)}",
        f"- no_overlap_safe radius overlap: {_yes_no(no_overlap_safe_has_overlap)}",
        f"- random seed: {config['project']['random_seed']}",
        f"- config_used.yaml: {config_used_path or (out / 'config_used.yaml')}",
        "",
        "## Window Counts",
        "",
        str(window_counts or {}),
        "",
        "## Plots",
        "",
        *plot_lines,
        "",
        "Safety distance thresholds are simulation parameters for controlled experiments, not normative construction-code values.",
    ]
    (out / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    enforce_quality_gates(stats, config)
    return stats
