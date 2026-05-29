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
python scripts/run_simulation.py --config configs/default.yaml --num-scenarios 100 --output-dir outputs/run_001
```

The script prints the output directory and key quality statistics after generation.

## Configuration

The YAML configuration controls:

- `project`: project name, version, random seed, and output directory
- `simulation`: scenario count, duration, timestep, crane-count range, save format
- `layout`: site size, overlap scene mix, and base-distance constraints
- `crane_static`: tower height, jib length, trolley radius limits, load capacity, safety-distance parameters
- `motion_limits`: angular/trolley/hoist speed and acceleration limits plus first-order response time
- `load_effect`: load-dependent acceleration scaling
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

Each run writes these files under the configured output directory:

- `tables/scenario_table.csv`: scenario id, scene type, crane count, duration, timestep, site size, seed, split
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
- `quality/plots/*.png`: diagnostic figures

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

Node inputs use `sin(theta)` and `cos(theta)` rather than raw `theta` to avoid angle wrap discontinuity.

## Data Leakage Rules

The implementation follows these rules:

- Future minimum distances and risk labels are never input features.
- `ttc_est_*` edge features are estimated from current distance and current relative approach speed only.
- Train/validation/test splits are made by `scenario_id`, never by random windows.
- `state_obs.csv` is the node-input source.
- `state_true.csv` and `edge_future_label.csv` are used for labels.

Run:

```bash
python -m pytest tests/test_no_future_leakage.py -q
```

## Quality Report

Inspect:

```bash
outputs/debug_small/quality/quality_report.md
```

The report includes scenario count, crane-count distribution, total simulated time, sampling frequency, task count, risk positive ratios, state distributions, distance distributions, NaN checks, boundary checks, velocity/acceleration checks, split exclusivity, leakage checks, seed, and `config_used.yaml` path.

Quality plots are written to:

```text
outputs/<run_name>/quality/plots/
```

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

CSV is the required baseline. Optional parquet support can be added by installing `pyarrow` and extending `io_utils.py`.

## Extension Ideas

- Load swing dynamics and pendulum-state labels
- Wind disturbance and sensor-fusion variants
- Jib deflection approximations under heavy load
- Load bounding volumes instead of point hooks
- Building/personnel/obstacle risk layers
- Unity, Gazebo, or Three.js visualization frontend
- Domain randomization for generalization tests
- Graph sparsification policies for large crane groups
