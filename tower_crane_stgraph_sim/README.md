# tower_crane_stgraph_sim

`tower_crane_stgraph_sim` is a reproducible Python dataset generator for the thesis topic:

> Physical-prior spatio-temporal dynamic graph trajectory prediction and collision-risk warning for multi tower-crane jib operations.

The simulator focuses on horizontal trolley tower cranes. Each crane is a graph node, and crane-pair geometry, relative motion, overlap, height difference, and current TTC estimates are physical-prior edge features. Future labels are computed from true simulated trajectories and include arm-arm, arm-hook, and hook-hook risk indicators.

This is a controllable data generator for ST-GNN/Transformer research. It is not a high-fidelity construction physics engine and does not model load swing, wind, jib deflection, physical load volume, personnel, buildings, Unity, Gazebo, ROS, or commercial crane-control systems. Safety distances are simulation parameters for experiments, not construction-code standards.

## Installation

Python 3.10+ is recommended.

```bash
cd tower_crane_stgraph_sim
python -m pip install -r requirements.txt
```

For editable development:

```bash
python -m pip install -e ".[test]"
```

## Quick Run

Small debug run:

```bash
python scripts/run_simulation.py --config configs/debug_small.yaml
```

Batch-style run:

```bash
python scripts/run_simulation.py --config configs/default.yaml --num-scenarios 20 --duration 600 --seed 42 --dt 0.2 --output-dir outputs/debug_run --run-id run_001
```

The script writes each generation under `project.output_dir/run_YYYYMMDD_HHMMSS/`. Pass `--run-id` when you want a reproducible run directory name. The script prints the resolved output directory and key quality statistics after generation.

`configs/default.yaml` is a medium default configuration. Use `configs/debug_small.yaml` or `configs/debug_fast.yaml` for quick checks, and `configs/full_large.yaml` for long-running 300/1000-scenario formal experiments.

## Configuration

The YAML configuration controls:

- `project`: project name, version, random seed, and output directory
- `simulation`: scenario count, duration, timestep, crane-count range, save format
- `layout`: site size, overlap scene mix, and base-distance constraints
- `crane_static`: tower height, jib length, trolley radius limits, load capacity, safety-distance parameters
- `motion_limits`: angular/trolley/hoist speed and acceleration limits plus first-order response time
- `load_effect`: load-dependent acceleration scaling
- `dynamics`: emergency braking scale, hook clearance, and other dynamic response parameters
- `task_generation`: pickup/dropoff heights, transport height, task count, load ratios, stage tolerance
- `controller`: proportional command gains and smoothing
- `interaction`: online short-horizon avoidance, failure/error probabilities, priority policy
- `risk_thresholds`: arm-arm, arm-hook, hook-hook risk thresholds
- `sensor_observation`: noise, bias, delay, dropout, and outlier settings
- `windowing`: input length, prediction horizon, stride, padding, max cranes
- `split`: scenario-level train/val/test split ratios
- `quality_control`: report and leakage-check settings

All random processes are controlled by `project.random_seed`.

## Output Tables

Each run writes these files under the generated run directory:

- `tables/scenario_table.csv`: scenario id/index, scene type, crane count, duration, timestep, site size, seed, split
- `tables/crane_static.csv`: static crane geometry and motion-limit parameters
- `tables/task_table.csv`: pickup/dropoff task definitions
- `tables/state_true.csv`: ground-truth simulated state and issued command fields
- `tables/state_obs.csv`: noisy/delayed/dropout observations used as model input
- `tables/geometry_table.csv`: reconstructed jib root/tip and hook coordinates
- `tables/edge_current.csv`: current-time physical-prior graph edge features
- `tables/edge_future_label.csv`: future minimum distances, risk labels, and label TTC values
- `windows/train_windows.npz`, `windows/val_windows.npz`, `windows/test_windows.npz`: model-ready sliding windows
- `windows/generalization_windows.npz`: optional generalization split when `split.add_generalization_test` is enabled
- `data_dictionary.md`: field definitions and input/label guidance
- `metadata.json`: generated-run metadata and split counts
- `quality/quality_report.md`: dataset statistics and integrity checks
- `quality/risk_ratio_by_scenario.csv`: per-scenario risk positive ratios
- `quality/feature_summary.csv`: numeric feature summary statistics for state, edge, and label tables
- `quality/plots/*.png`: diagnostic figures

Tables use stable string business IDs such as `scenario_000000`, `crane_00`, and `task_00_0000` in `scenario_id`, `crane_id`, and `task_id`. Numeric `*_index` fields are used for seeding, splitting, joins, and tensor construction. `scenario_uid`, `crane_uid`, and `task_uid` are deprecated aliases retained for compatibility.

## Sliding Window Tensors

The npz files contain:

- `node_features`: `[num_samples, T_in, N_max, F_node]`
- `edge_features`: `[num_samples, T_in, N_max, N_max, F_edge]`
- `node_mask`: `[num_samples, N_max]`
- `edge_mask`: `[num_samples, N_max, N_max]`
- `time_mask`: `[num_samples, T_in]`
- `y_traj`: `[num_samples, H, N_max, 4]`
- `y_risk`: `[num_samples, N_max, N_max, 4]`
- `y_min_distance`: `[num_samples, N_max, N_max, 4]`

Feature-name arrays are stored in every npz:

- `node_feature_names`
- `edge_feature_names`
- `y_traj_feature_names`
- `y_risk_feature_names`
- `y_min_distance_feature_names`

Window files include string business `scenario_ids`, integer `scenario_indices`, and compatibility aliases `scenario_business_ids` and `scenario_uids` for schema-level traceability.

Node inputs use `sin(theta)` and `cos(theta)` rather than raw `theta` to avoid angle wrap discontinuity.

## Data Leakage Rules

The implementation follows these rules:

- Future minimum distances and risk labels are never input features.
- `ttc_est_*` edge features are estimated from current distance and current-state constant-velocity extrapolated approach speed only.
- Train/validation/test splits are made by `scenario_id`, never by random windows.
- `state_obs.csv` is the node-input source.
- `state_true.csv` and `edge_future_label.csv` are used for labels.
- Incomplete tail horizons are skipped, so `edge_future_label.csv` should not contain `inf` values from missing future windows.

Run:

```bash
python -m pytest tests/test_no_future_leakage.py -q
```

## Quality Report

Inspect:

```bash
outputs/debug_small/run_YYYYMMDD_HHMMSS/quality/quality_report.md
```

The report includes scenario count, crane-count distribution, total simulated time, sampling frequency, task count, risk positive ratios, state distributions, distance distributions, NaN checks, boundary checks, velocity/acceleration checks, all-split exclusivity including optional generalization windows, leakage checks, tail-label `inf` checks, no-overlap consistency checks, seed, and `config_used.yaml` path. The generator also writes `risk_ratio_by_scenario.csv` and `feature_summary.csv` in the same quality directory.

`same_height_risk_zone` is an edge feature indicating close jib-root heights together with plan-view operating-radius overlap. It is not a universal height-risk label for arm-hook or hook-hook risk.

Quality plots are written to:

```text
outputs/<base_dir>/<run_name>/quality/plots/
```

## 可视化界面

仓库包含一个基于 Streamlit 的浏览器可视化验收界面，以及位于 `tower_sim.visualization` 下的可测试后端模块。后端既能加载正式 run 布局（`tables/`、`windows/`、`quality/`），也兼容旧版平铺的 `outputs/debug_small` 布局。

安装可选界面依赖：

```bash
python -m pip install -e ".[visual]"
```

启动可视化界面：

```bash
streamlit run apps/visual_dashboard.py
```

界面包含运行总览、动画播放器、风险解释器、训练窗口查看器、质量视图和数据浏览器。输入视图只使用 `state_obs` 和 `edge_current`；标签视图和调试视图会展示 `edge_future_label`，但这些内容只作为标签验收证据，不能作为模型输入。

如果需要非交互导出或自动化检查，可以直接使用这些 Python 模块：

- `tower_sim.visualization.loaders.RunDataRepository`
- `tower_sim.visualization.animation_player`
- `tower_sim.visualization.risk_events`
- `tower_sim.visualization.window_view`
- `tower_sim.visualization.quality_view`
- `tower_sim.visualization.export`

## Using With ST-GNN or Transformer Models

A typical model pipeline is:

1. Load one split npz.
2. Use `node_features`, `edge_features`, `node_mask`, `edge_mask`, and `time_mask` as inputs.
3. Predict `y_traj` for each crane.
4. Use predicted trajectories to reconstruct future jib segments and hook points.
5. Compute pairwise future distances and compare against `y_risk` / `y_min_distance`.
6. Train multi-task losses, for example trajectory regression plus risk classification.

The edge tensor is dense padded. `edge_mask` indicates valid crane pairs.

## Tests

```bash
python -m pytest -q
```

The tests cover geometry distance functions, dynamic bounds, future-label behavior, and leakage rules.

## FAQ

**Is this a construction safety standard implementation?**

No. Safety thresholds are configurable experiment parameters only.

**Does it model load swing or wind?**

No. The baseline intentionally excludes swing, wind, jib deflection, and load volume to keep the dataset generator controlled and interpretable.

**Why use `sin(theta)` and `cos(theta)`?**

They avoid discontinuity around `0` and `2*pi`.

**Can I generate parquet instead of CSV?**

Yes. Set `simulation.save_format` to include `parquet`. Downstream scripts read CSV first and fall back to Parquet when CSV is absent. `simulation.save_format` must include at least one table format, `csv` or `parquet`; `npz` only documents that window output is part of the run and cannot be used as the sole format.

**Does `emergency_flag` change the motion?**

Yes. Emergency commands zero the requested velocities and use `dynamics.emergency_brake_scale` to apply stronger deceleration toward a stop.

## Extension Ideas

- Load swing dynamics and pendulum-state labels
- Wind disturbance and sensor-fusion variants
- Jib deflection approximations under heavy load
- Load bounding volumes instead of point hooks
- Building/personnel/obstacle risk layers
- Unity, Gazebo, or Three.js visualization frontend
- Domain randomization for generalization tests
- Graph sparsification policies for large crane groups
