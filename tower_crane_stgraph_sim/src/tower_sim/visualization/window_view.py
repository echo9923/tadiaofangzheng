from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tower_sim.visualization.schemas import RISK_TYPE_SPECS, assert_input_columns_safe, future_label_offenders


@dataclass(frozen=True)
class WindowBundle:
    split: str
    path: Path | None
    arrays: dict[str, np.ndarray]

    @property
    def num_samples(self) -> int:
        if "node_features" not in self.arrays:
            return 0
        return int(self.arrays["node_features"].shape[0])


@dataclass(frozen=True)
class WindowSampleView:
    split: str
    sample_index: int
    scenario_id: Any
    scenario_index: int | None
    window_start_step: int | None
    tensors: dict[str, np.ndarray]
    shapes: dict[str, tuple[int, ...]]
    masks: dict[str, np.ndarray]
    risk_positive_count: int


def load_window_bundle(path: str | Path, split: str | None = None) -> WindowBundle:
    path_obj = Path(path)
    with np.load(path_obj, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    inferred_split = split or path_obj.name.removesuffix("_windows.npz")
    return WindowBundle(inferred_split, path_obj, arrays)


def bundle_from_arrays(arrays: dict[str, np.ndarray], split: str = "memory") -> WindowBundle:
    return WindowBundle(split=split, path=None, arrays={key: np.asarray(value) for key, value in arrays.items()})


def _names(arrays: dict[str, np.ndarray], key: str) -> list[str]:
    if key not in arrays:
        return []
    return [str(value) for value in arrays[key].tolist()]


def check_input_feature_leakage(bundle: WindowBundle) -> list[str]:
    node_offenders = future_label_offenders(_names(bundle.arrays, "node_feature_names"))
    edge_offenders = future_label_offenders(_names(bundle.arrays, "edge_feature_names"))
    return [f"node:{name}" for name in node_offenders] + [f"edge:{name}" for name in edge_offenders]


def assert_window_inputs_safe(bundle: WindowBundle) -> None:
    assert_input_columns_safe(_names(bundle.arrays, "node_feature_names"), context=f"{bundle.split} node_features")
    assert_input_columns_safe(_names(bundle.arrays, "edge_feature_names"), context=f"{bundle.split} edge_features")


def summarize_window_bundle(bundle: WindowBundle) -> dict[str, Any]:
    arrays = bundle.arrays
    summary: dict[str, Any] = {
        "split": bundle.split,
        "path": str(bundle.path) if bundle.path is not None else None,
        "num_samples": bundle.num_samples,
        "shapes": {key: tuple(value.shape) for key, value in arrays.items() if hasattr(value, "shape")},
        "node_feature_names": _names(arrays, "node_feature_names"),
        "edge_feature_names": _names(arrays, "edge_feature_names"),
        "leakage_offenders": check_input_feature_leakage(bundle),
    }
    if "scenario_ids" in arrays:
        summary["scenario_count"] = int(len(set(arrays["scenario_ids"].tolist())))
    if "y_risk" in arrays and arrays["y_risk"].size:
        summary["risk_positive_samples"] = int((arrays["y_risk"].max(axis=(1, 2, 3)) > 0).sum())
    else:
        summary["risk_positive_samples"] = 0
    return summary


def _sample_meta(arrays: dict[str, np.ndarray], index: int) -> tuple[Any, int | None, int | None]:
    scenario_id = arrays.get("scenario_ids", np.array([None] * (index + 1)))[index]
    scenario_index = None
    if "scenario_indices" in arrays and len(arrays["scenario_indices"]) > index:
        scenario_index = int(arrays["scenario_indices"][index])
    window_start_step = None
    if "window_start_steps" in arrays and len(arrays["window_start_steps"]) > index:
        window_start_step = int(arrays["window_start_steps"][index])
    return scenario_id, scenario_index, window_start_step


def get_window_sample(bundle: WindowBundle, sample_index: int) -> WindowSampleView:
    if sample_index < 0 or sample_index >= bundle.num_samples:
        raise IndexError(f"sample_index out of range for {bundle.split}: {sample_index}")
    tensors: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for key in ["node_features", "edge_features", "y_traj", "y_risk", "y_min_distance"]:
        if key in bundle.arrays:
            tensors[key] = bundle.arrays[key][sample_index]
    for key in ["node_mask", "edge_mask", "time_mask"]:
        if key in bundle.arrays:
            masks[key] = bundle.arrays[key][sample_index]
    scenario_id, scenario_index, window_start_step = _sample_meta(bundle.arrays, sample_index)
    risk_positive_count = int(tensors.get("y_risk", np.zeros(0)).sum())
    return WindowSampleView(
        split=bundle.split,
        sample_index=sample_index,
        scenario_id=scenario_id,
        scenario_index=scenario_index,
        window_start_step=window_start_step,
        tensors=tensors,
        shapes={key: tuple(value.shape) for key, value in {**tensors, **masks}.items()},
        masks=masks,
        risk_positive_count=risk_positive_count,
    )


def filter_window_samples(
    bundle: WindowBundle,
    *,
    scenario_id: Any | None = None,
    positive_only: bool = False,
    risk_type: str | None = None,
    contains_brake: bool = False,
    contains_emergency: bool = False,
    contains_missing: bool = False,
) -> list[int]:
    indices = list(range(bundle.num_samples))
    arrays = bundle.arrays
    if scenario_id is not None and "scenario_ids" in arrays:
        indices = [idx for idx in indices if str(arrays["scenario_ids"][idx]) == str(scenario_id)]
    if positive_only and "y_risk" in arrays:
        indices = [idx for idx in indices if arrays["y_risk"][idx].max() > 0]
    if risk_type is not None and "y_risk" in arrays and "y_risk_feature_names" in arrays:
        risk_names = _names(arrays, "y_risk_feature_names")
        spec = RISK_TYPE_SPECS[risk_type]
        if spec.risk_column in risk_names:
            risk_idx = risk_names.index(spec.risk_column)
            indices = [idx for idx in indices if arrays["y_risk"][idx, :, :, risk_idx].max() > 0]
    if (contains_brake or contains_emergency or contains_missing) and "node_features" in arrays and "node_feature_names" in arrays:
        names = _names(arrays, "node_feature_names")

        def has_feature(idx: int, feature: str) -> bool:
            return feature in names and bool(np.nanmax(arrays["node_features"][idx, :, :, names.index(feature)]) > 0)

        if contains_brake:
            indices = [idx for idx in indices if has_feature(idx, "brake_flag")]
        if contains_emergency:
            indices = [idx for idx in indices if has_feature(idx, "emergency_flag")]
        if contains_missing:
            missing_names = [name for name in names if name.startswith("missing_mask_")]
            indices = [
                idx
                for idx in indices
                if any(bool(np.nanmax(arrays["node_features"][idx, :, :, names.index(name)]) > 0) for name in missing_names)
            ]
    return indices
