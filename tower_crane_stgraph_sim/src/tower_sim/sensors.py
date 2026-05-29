from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OBS_COLUMNS = [
    "theta",
    "r",
    "h",
    "theta_dot",
    "r_dot",
    "h_dot",
    "theta_ddot",
    "r_ddot",
    "h_ddot",
    "load_weight",
    "command_theta",
    "command_r",
    "command_h",
    "brake_flag",
    "emergency_flag",
]


def generate_observations(state_true: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Generate noisy, delayed, dropout-prone observations from true states."""

    sensor_cfg = config["sensor_observation"]
    rows: list[pd.DataFrame] = []
    enabled = bool(sensor_cfg.get("enabled", True))
    for (_, crane_id), group in state_true.sort_values(["scenario_id", "crane_id", "step"]).groupby(["scenario_id", "crane_id"]):
        g = group.copy().reset_index(drop=True)
        delay_min, delay_max = [int(x) for x in sensor_cfg.get("delay_steps_range", [0, 0])]
        delays = rng.integers(delay_min, delay_max + 1, size=len(g)) if enabled else np.zeros(len(g), dtype=int)
        obs = g.copy()
        for idx, delay in enumerate(delays):
            src_idx = max(0, idx - int(delay))
            for col in OBS_COLUMNS:
                obs.loc[idx, col] = g.loc[src_idx, col]
        if enabled:
            theta_bias = rng.normal(0.0, float(sensor_cfg["theta_bias_std_rad"]))
            r_bias = rng.normal(0.0, float(sensor_cfg["r_bias_std_m"]))
            h_bias = rng.normal(0.0, float(sensor_cfg["h_bias_std_m"]))
            obs["theta"] += theta_bias + rng.normal(0.0, float(sensor_cfg["theta_noise_std_rad"]), size=len(obs))
            obs["r"] += r_bias + rng.normal(0.0, float(sensor_cfg["r_noise_std_m"]), size=len(obs))
            obs["h"] += h_bias + rng.normal(0.0, float(sensor_cfg["h_noise_std_m"]), size=len(obs))

        obs["theta_missing"] = 0
        obs["r_missing"] = 0
        obs["h_missing"] = 0
        obs["is_outlier"] = 0
        if enabled:
            dropout_probability = float(sensor_cfg.get("dropout_probability", 0.0))
            outlier_probability = float(sensor_cfg.get("outlier_probability", 0.0))
            for col, mask_col in [("theta", "theta_missing"), ("r", "r_missing"), ("h", "h_missing")]:
                mask = rng.random(len(obs)) < dropout_probability
                obs.loc[mask, mask_col] = 1
                if bool(sensor_cfg.get("keep_missing_mask", True)):
                    obs.loc[mask, col] = np.nan
            outlier_mask = rng.random(len(obs)) < outlier_probability
            obs.loc[outlier_mask, "is_outlier"] = 1
            obs.loc[outlier_mask, "theta"] += rng.normal(0.0, 0.5, size=int(outlier_mask.sum()))
            obs.loc[outlier_mask, "r"] += rng.normal(0.0, 5.0, size=int(outlier_mask.sum()))
            obs.loc[outlier_mask, "h"] += rng.normal(0.0, 5.0, size=int(outlier_mask.sum()))
        obs["obs_delay_steps"] = delays.astype(int)
        rows.append(obs)
    result = pd.concat(rows, ignore_index=True) if rows else state_true.copy()
    return result[
        [
            "scenario_id",
            "timestamp",
            "step",
            "crane_id",
            *OBS_COLUMNS,
            "task_id",
            "task_stage",
            "theta_missing",
            "r_missing",
            "h_missing",
            "obs_delay_steps",
            "is_outlier",
        ]
    ]
