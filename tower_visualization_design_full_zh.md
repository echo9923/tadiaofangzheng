# 群塔吊时空图仿真数据可视化系统完整设计方案

**Animation Player + Dashboard + Risk Inspector + Window Explorer + Quality View**

- 适用项目：`tower_crane_stgraph_sim / 群塔起重臂轨迹预测与碰撞风险预警数据集`
- 版本：v1.0
- 日期：2026-06-01
- 设计依据：用户提供的《AI Prompt 方案：群塔完整版仿真数据生成器》以及当前项目输出结构。

# 0. 文档定位与使用方式

本方案是一份面向工程实现和论文验收的完整可视化系统设计文档。它将“群塔吊仿真数据生成器”的输出结果转化为可播放、可解释、可检查、可导出的交互式可视化系统。系统核心不是简单动画展示，而是围绕 ST-GNN/Transformer 数据集生成流程，验证节点特征、边特征、未来标签、滑动窗口、split 和质量报告是否正确。

| 文档部分 | 解决的问题 | 主要产出 |
| --- | --- | --- |
| 技术设计 | 如何实现、如何加载数据、如何播放动画、如何做性能优化 | 模块结构、数据仓库、缓存、动画帧、前端方案 |
| 流程设计 | 用户从选择 run 到解释风险、检查窗口、导出材料的路径 | 页面流、数据流、交互流、风险事件流 |
| 任务设计 | 研究者、模型开发者、展示人员分别如何使用系统 | 任务清单、用户故事、典型操作路径 |
| 验收标准 | 如何判断可视化系统是否合格 | 功能验收、动画验收、无泄漏验收、性能验收、质量验收 |

推荐先实现 Streamlit 版 MVP，确保 debug_small 输出可以完整播放和检查；随后增加风险事件回放、窗口样本检查、质量报告交互化和导出功能。

# 1. 系统定位

## 1.1 系统名称

建议命名为：TowerSim Visual Dashboard。中文名可以使用“群塔吊时空图仿真数据可视化与质量验收系统”。

## 1.2 一句话定位

本系统是一个面向群塔吊 ST-GNN/Transformer 数据集的可视化验收平台：它读取仿真器已经生成的数据表和窗口 NPZ，按 scenario 和 step 播放群塔吊运动动画，同时解释当前边特征、未来风险标签、训练窗口和质量报告。

## 1.3 系统目标

- 播放多塔吊 scenario 的 2.5D 动画，展示塔吊基座、作业半径、吊臂、小车、吊钩、任务点和风险边。
- 区分 Input View、Label View 和 Debug View，防止未来标签被误认为模型输入。
- 可解释 risk_* 标签：展示未来窗口内的最小距离、首次进入阈值时间和距离曲线。
- 可检查 ST-GNN/Transformer 训练窗口：node_features、edge_features、mask、y_traj、y_risk、y_min_distance。
- 可验收数据质量：风险比例、NaN/inf、越界、速度/加速度超限、split 互斥、future leakage。
- 可导出论文和答辩材料：高分辨率 PNG、风险解释图、GIF/MP4 片段、Markdown 说明。

## 1.4 不做什么

- 不重新运行物理仿真；只读取已有输出。
- 不替代模型训练；只检查和解释训练数据。
- 不作为施工现场实时安全监控系统。
- 不优先实现 Unity/Gazebo 高保真三维仿真；第一阶段采用 2.5D 数据回放动画。

# 2. 总体功能架构

## 2.1 页面结构

系统建议包含 7 个主页面，其中 Animation Player 是核心页面。

| 页面 | 中文名称 | 核心用途 |
| --- | --- | --- |
| Run Dashboard | 运行总览页 | 选择 run，查看整体数据质量和配置摘要 |
| Animation Player | 群塔动画播放器 | 按 scenario/step 播放塔吊运动，显示当前边和风险边 |
| Risk Inspector | 风险解释器 | 解释某个 risk=1 的来源、TTC、未来最小距离和阈值 |
| Window Explorer | 训练窗口查看器 | 检查 NPZ 样本、张量 shape、mask、标签对齐和无泄漏 |
| Quality View | 质量报告页 | 交互化展示 quality_report、risk_ratio、feature_summary 和异常场景 |
| Data Browser | 数据表浏览器 | 按 scenario/crane/step 筛选查看原始表 |
| Export Center | 导出中心 | 导出 PNG、GIF/MP4、Markdown、CSV 和报告包 |

## 2.2 页面关系

```text
Run Dashboard
  -> Animation Player
       -> Risk Inspector
       -> Window Explorer
       -> Export Center
  -> Quality View
       -> Data Browser
       -> Animation Player
  -> Data Browser
```

## 2.3 核心原则

- 以 scenario 为最小分析单元，所有回放、风险解释和 split 检查都应围绕 scenario_id 展开。
- 动画播放基于 geometry_table/state/edge 表逐帧回放，不是实时重跑仿真。
- 输入和标签视觉隔离：Input View 不能读取或显示 edge_future_label。
- 质量验收优先于动画美观；动画要能帮助发现数据问题。
- 先做 2.5D 俯视图 + 高度剖面，再考虑 Three.js/Unity。

# 3. 数据输入与文件契约

## 3.1 必需文件

| 文件 | 用途 | 缺失时处理 |
| --- | --- | --- |
| metadata.json | run 元信息、生成时间、版本、配置摘要 | Dashboard 显示错误 |
| config_used.yaml | 复现实验配置、阈值、窗口长度、速度限制 | Dashboard 显示错误 |
| tables/scenario_table | 场景列表、scene_type、split、num_cranes | 无法启动主要页面 |
| tables/crane_static | 基座、塔高、臂长、作业半径、运动限制 | Animation Player 无法绘制静态元素 |
| tables/task_table | 任务点、任务开始时间、载荷、priority | 动画可播放，但任务图层不可用 |
| tables/state_true | 真实状态、标签验证、质量检查 | Label View 和质量检查不可用 |
| tables/state_obs | 模型输入视图、传感器噪声、missing/outlier | Input View 不可用 |
| tables/geometry_table | root/tip/hook，动画几何基础 | Animation Player 不可用 |
| tables/edge_current | 当前边距离、TTC、overlap、相对运动 | 当前边图层不可用 |
| tables/edge_future_label | 未来风险标签、未来最小距离、TTC label | Label View 和 Risk Inspector 不可用 |

## 3.2 可选文件

| 文件 | 用途 |
| --- | --- |
| windows/train_windows.npz | 查看训练集窗口样本 |
| windows/val_windows.npz | 查看验证集窗口样本 |
| windows/test_windows.npz | 查看测试集窗口样本 |
| windows/generalization_windows.npz | 查看泛化测试窗口样本 |
| quality/quality_report.md | 质量报告原文展示 |
| quality/risk_ratio_by_scenario.csv | 场景风险比例分布和异常场景筛选 |
| quality/feature_summary.csv | 特征统计摘要 |
| quality/plots/*.png | 质量图展示和论文导出 |

## 3.3 CSV/Parquet 兼容

所有表读取函数应优先读取 Parquet，若不存在则读取 CSV；两者都不存在时返回清晰的缺失文件错误。

```text
def read_table(tables_dir, name):
    parquet_path = tables_dir / f"{name}.parquet"
    csv_path = tables_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing table: {name}")
```

# 4. 技术设计

## 4.1 第一阶段技术栈

| 层级 | 推荐技术 | 原因 |
| --- | --- | --- |
| 界面层 | Streamlit | 开发快，适合论文调试、截图和交互式数据浏览 |
| 绘图层 | Plotly | 支持交互式 scatter/line/animation、悬停提示和导出 |
| 数据层 | pandas + numpy | 直接处理 CSV/Parquet/NPZ |
| 配置层 | pyyaml | 读取 config_used.yaml 和 visualization.yaml |
| 报告导出 | kaleido / matplotlib / imageio | 导出 PNG、GIF 或 MP4 |

## 4.2 第二阶段扩展技术栈

| 目标 | 可选技术 | 适用情况 |
| --- | --- | --- |
| 更流畅动画 | Dash + Plotly | 需要更好的浏览器端播放和回调控制 |
| 前后端分离 | FastAPI + React/Vue | 后续要做正式系统演示 |
| 高性能绘图 | Canvas/WebGL/Three.js | 场景很多、帧率要求高、需要 3D 透视 |
| 大数据查询 | DuckDB + Parquet 分区 | 正式数据集体量较大时 |

## 4.3 推荐代码目录

```text
tower_crane_stgraph_sim/
  apps/
    visual_dashboard.py
  src/tower_sim/visualization/
    __init__.py
    schemas.py
    loaders.py
    indexing.py
    frame_cache.py
    plotting.py
    animation_player.py
    risk_events.py
    window_view.py
    quality_view.py
    export.py
    ui_state.py
  configs/
    visualization.yaml
  tests/
    test_visualization_loaders.py
    test_visualization_risk_events.py
    test_visualization_window_view.py
    test_visualization_no_leakage_ui.py
```

## 4.4 核心模块职责

| 模块 | 职责 |
| --- | --- |
| schemas.py | 定义表字段、特征字段、必需列和无泄漏字段黑名单 |
| loaders.py | 读取 run_root、CSV/Parquet、NPZ、metadata、config，并做缓存 |
| indexing.py | 构建 scenario index、risk event index、window sample index |
| frame_cache.py | 按 scenario 和 step 构建动画帧，避免重复读盘 |
| plotting.py | 绘制 2.5D 俯视图、高度剖面、距离曲线和时间序列 |
| animation_player.py | 封装播放/暂停/跳转/风险片段播放等 UI 逻辑 |
| risk_events.py | 从 edge_future_label 提取风险事件并生成解释文本 |
| window_view.py | 解析 NPZ、显示 shape、mask、特征列表和样本回放 |
| quality_view.py | 解析质量报告、风险比例、feature summary 和异常场景 |
| export.py | 导出 PNG、GIF/MP4、Markdown、CSV 和风险解释包 |

# 5. 数据仓库与缓存设计

## 5.1 RunDataRepository

建议建立统一的数据仓库类，所有页面都通过它读取数据，避免每个页面重复处理文件路径、CSV/Parquet 兼容、缓存和错误提示。

```text
class RunDataRepository:
    def __init__(self, run_root: Path):
        self.run_root = run_root
        self.tables_dir = run_root / "tables"
        self.windows_dir = run_root / "windows"
        self.quality_dir = run_root / "quality"

    def load_metadata(self) -> dict: ...
    def load_config(self) -> dict: ...
    def load_table(self, name: str, columns: list[str] | None = None) -> pd.DataFrame: ...
    def load_scenario_table(self) -> pd.DataFrame: ...
    def load_scenario_data(self, scenario_id: str) -> ScenarioData: ...
    def load_window_npz(self, split: str) -> dict[str, np.ndarray]: ...
```

## 5.2 ScenarioData

```text
@dataclass
class ScenarioData:
    scenario_id: str
    scenario_row: pd.Series
    crane_static: pd.DataFrame
    task_table: pd.DataFrame
    state_true: pd.DataFrame
    state_obs: pd.DataFrame
    geometry: pd.DataFrame
    edge_current: pd.DataFrame
    edge_future_label: pd.DataFrame | None
```

## 5.3 动画帧对象

```text
@dataclass
class AnimationFrame:
    scenario_id: str
    step: int
    timestamp: float
    state: pd.DataFrame
    geometry: pd.DataFrame
    edges: pd.DataFrame
    labels: pd.DataFrame | None
    tasks: pd.DataFrame
    static: pd.DataFrame
```

## 5.4 懒加载原则

- 启动时只读取 metadata、config、scenario_table、quality summary。
- 用户选择 scenario 后，再读取该 scenario 的 state/geometry/edge/label。
- 用户选择 split 和 sample 后，再读取对应 NPZ。
- 正式数据集优先读取 Parquet，长曲线绘图时做降采样。
- 动画播放缓存当前 scenario 的常用帧，但不要一次缓存所有 scenario。

# 6. Animation Player 动画播放器设计

## 6.1 定位

Animation Player 是系统核心。它按 scenario_id 和 step 逐帧读取 geometry_table、state_true/state_obs、edge_current 和可选的 edge_future_label，播放多塔吊 2.5D 动画。它不是重跑仿真，而是对已有仿真结果进行数据回放。

## 6.2 页面布局

```text
顶部：Run / Scenario / View Mode / Horizon / 播放速度
左侧：scenario 筛选、风险类型、图层开关、塔吊选择、塔吊对选择
中间：2.5D 动画画布，显示塔吊、吊臂、吊钩、任务点、边和风险边
右侧：选中塔吊详情、选中边详情、任务阶段、当前/未来风险信息
底部：时间轴、播放按钮、上一帧/下一帧、风险事件 marker、高度剖面图
```

## 6.3 播放控制

- 播放、暂停、重播。
- 上一帧、下一帧。
- 跳转到 step 或 timestamp。
- 拖动时间轴。
- 播放速度：0.5x、1x、2x、5x、10x。
- 循环播放当前 scenario。
- 跳转到下一个风险事件、上一个风险事件。
- 播放风险事件前后片段，例如事件前 10 秒到后 10 秒。

## 6.4 视图模式

| 模式 | 允许显示 | 禁止显示 | 用途 |
| --- | --- | --- | --- |
| Input View | state_obs、edge_current、当前距离、ttc_est、missing/outlier、历史轨迹 | edge_future_label、risk_*、future_min_d_*、ttc_label_*、未来真实轨迹 | 模拟模型真实输入 |
| Label View | state_true、edge_future_label、未来最小距离、risk、ttc_label、未来真实轨迹 | 不得暗示这些是模型输入 | 验证标签正确性 |
| Debug View | 输入与标签同时显示，但要明显区分历史/当前/未来 | 不得用于训练输入定义 | 论文解释和调试 |

## 6.5 2.5D 主画布元素

| 元素 | 数据来源 | 显示方式 |
| --- | --- | --- |
| 塔吊基座 | crane_static.base_x/base_y | 圆点 + crane_id + priority |
| 最大作业半径 | crane_static.max_radius | 半透明圆 |
| 最小作业半径 | crane_static.min_radius | 虚线圆 |
| 吊臂 | geometry.root_x/y 到 tip_x/y | 线段，随 theta 旋转 |
| 小车/吊钩水平位置 | geometry.hook_x/hook_y | 点，旁边显示 h |
| 任务点 | task_table pickup/dropoff 转换到 x-y | pickup 三角形，dropoff 方形 |
| 当前边 | edge_current | 安全灰色、接近橙色、当前风险红色 |
| 未来风险边 | edge_future_label | 红色或紫色虚线，仅 Label/Debug View |

## 6.6 高度辅助图

俯视图无法表达 h 和塔高差，因此动画播放器底部或右侧应增加选中塔吊对的高度剖面图。它显示 tower_height、root_z、hook_z、delta_h 和 delta_tower_height。此图用于解释“俯视图接近但高度错开”的情况。

## 6.7 时间轴事件 marker

| 事件 | 来源 | 显示 |
| --- | --- | --- |
| 任务阶段切换 | state_true.task_stage 或 task_table | 灰色竖线 |
| brake_flag | state_true/state_obs.brake_flag | 黄色圆点 |
| emergency_flag | state_true/state_obs.emergency_flag | 红色圆点 |
| 未来风险事件 | edge_future_label.risk_* | 红色三角 |
| outlier | state_obs.is_outlier | 蓝色叉号 |
| missing | state_obs.*_missing | 空心点或淡化标记 |

## 6.8 风险片段播放

风险片段播放是动画播放器的关键功能。用户选择某个 risk event 后，系统自动定位到事件 step，并播放 [event_step - pre_seconds, event_step + post_seconds] 范围。默认 pre_seconds=10，post_seconds=10。

```text
Risk clip example:
scenario_id = scenario_000023
crane_i = crane_01
crane_j = crane_04
risk_type = arm_hook_i_to_j
event_step = 1840
play_range = [1790, 1890]  # dt=0.2, roughly +/-10s
```

# 7. Run Dashboard 设计

## 7.1 目标

Run Dashboard 用来判断一次仿真 run 是否整体合格。用户应能在一个页面内看到场景数量、总时长、风险比例、质量 gate、split 分布和异常 scenario。

## 7.2 指标卡片

- Run ID、配置文件、随机种子、生成时间。
- num_scenarios、total_effective_hours、dt、sampling_rate。
- num_cranes 分布、scene_type 分布、split 分布。
- risk_any_ratio 和三类风险比例。
- NaN/inf、r/h 越界、速度/加速度越界。
- future leakage、split disjoint、quality gate 总状态。

## 7.3 图表

- scene_type 数量柱状图。
- split 数量柱状图。
- num_cranes 分布图。
- risk_ratio_by_scenario 分布图。
- 按 scene_type 和 split 分组的平均风险比例。
- 异常 scenario 列表，点击可跳转到 Animation Player。

# 8. Risk Inspector 风险解释器设计

## 8.1 目标

Risk Inspector 用于解释某个风险标签为什么为 1，或者为什么某个当前看似接近的塔吊对没有未来风险。它将当前输入窗口、未来标签窗口、距离曲线、阈值线、TTC 估计和 TTC 标签放到同一页面中解释。

## 8.2 风险事件索引

从 edge_future_label 中提取 risk=1 的事件，并将四类风险拆成独立事件：arm_arm、arm_hook_i_to_j、arm_hook_j_to_i、hook_hook。

| 字段 | 说明 |
| --- | --- |
| scenario_id / step / timestamp | 风险标签对应的当前输入窗口末端 |
| horizon_s | 预测窗口长度 |
| crane_i / crane_j | 发生风险的塔吊对 |
| risk_type | AA/AH/HA/HH 之一 |
| future_min_distance | 未来窗口内该风险类型的最小距离 |
| ttc_label | 未来首次进入风险阈值的时间，无风险为 -1 |

## 8.3 距离曲线与窗口背景

对于选中 pair，绘制 d_arm_arm、d_arm_hook_i_to_j、d_arm_hook_j_to_i、d_hook_hook 随时间变化曲线，并添加安全阈值线。背景应明确标出历史输入窗口 [t-T+1, t] 和未来标签窗口 [t+1, t+H]。

## 8.4 自动解释文本

```text
示例：
在 scenario_000023 的 t=368.0s，crane_01 与 crane_04 在 8.0s 预测窗口内出现 arm-hook 风险。
当前时刻 d_arm_hook_i_to_j=4.28m，尚未低于安全阈值 2.0m。
未来窗口内最小距离为 1.37m，发生在 t+5.2s。
因此 risk_arm_hook_i_to_j=1，ttc_label_arm_hook=5.2s。
该标签来自 state_true 的未来轨迹，不属于模型输入。
```

# 9. Window Explorer 训练窗口查看器设计

## 9.1 目标

Window Explorer 用于检查 .npz 训练样本。它不是普通数据表浏览器，而是要确认 ST-GNN/Transformer 的 node_features、edge_features、mask、y_traj、y_risk、y_min_distance 是否正确对齐。

## 9.2 必须展示的张量

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| node_features | [T_in, N_max, F_node] | 历史输入节点特征 |
| edge_features | [T_in, N_max, N_max, F_edge] | 历史输入边特征 |
| y_traj | [H, N_max, F_y] | 未来轨迹标签 |
| y_risk | [N_max, N_max, F_risk] | 未来风险标签 |
| y_min_distance | [N_max, N_max, F_dist] | 未来最小距离标签 |
| node_mask | [N_max] | 有效塔吊 mask |
| edge_mask | [N_max, N_max] | 有效边 mask |
| time_mask | [T_in] | 有效时间 mask |

## 9.3 样本筛选

- 按 split：train、val、test、generalization。
- 按 scenario_id。
- 只看 y_risk 有正样本的窗口。
- 只看 arm_arm / arm_hook / hook_hook 风险窗口。
- 只看包含 emergency 或 brake 的窗口。
- 只看存在 missing/outlier 的窗口。

## 9.4 无泄漏检查

Window Explorer 必须自动检查 edge_features 和 node_features 的字段名，确认不包含 future_min_d、risk、ttc_label 等未来标签字段。

```text
Forbidden input feature substrings:
- future_min_d
- risk_
- ttc_label
- label
- future
```

# 10. Quality View 质量报告页设计

## 10.1 目标

Quality View 负责把 quality_report.md、risk_ratio_by_scenario.csv、feature_summary.csv 和 plots 交互化展示，让用户快速判断数据是否能用于正式训练和论文实验。

## 10.2 质量指标

| 指标 | 正式建议标准 | 说明 |
| --- | --- | --- |
| num_scenarios | >= 100 | 正式训练数据规模 |
| num_cranes_range | [3, 6] | 群塔场景 |
| total_effective_hours | >= 50 | 有效仿真总时长 |
| risk_positive_ratio | 0.10 - 0.30 | 风险正样本比例 |
| future_leakage | False | 无未来标签进入输入 |
| NaN/inf | 必需字段无 NaN/inf | 标签和必需输入字段有效 |
| bounds | 无 r/h 越界 | 物理状态合法 |
| speed/acc | 无超限或符合制动放宽逻辑 | 运动限制合法 |
| split | scenario_id 互斥 | train/val/test/generalization 不混场景 |

## 10.3 异常场景列表

- 风险比例过低或过高的 scenario。
- 存在 NaN/inf 的 scenario。
- 存在越界、超速、超加速度的 scenario。
- 几乎全 idle 的 scenario。
- 含大量 emergency 或 avoidance failure 的 scenario。
- no_overlap_safe 中发生半径重叠的 scenario。

# 11. Data Browser 与 Export Center

## 11.1 Data Browser

Data Browser 用于直接筛选和查看原始表。必须支持按 scenario_id、crane_id、step/timestamp、crane_i/crane_j、risk_type 过滤。打开 edge_future_label 时必须显示“未来标签，不可作为模型输入”的提示。

## 11.2 Export Center

| 导出内容 | 格式 | 用途 |
| --- | --- | --- |
| 当前动画帧 | PNG/SVG/PDF | 论文图和答辩截图 |
| 风险片段动画 | GIF/MP4 | 展示风险形成过程 |
| 风险解释包 | PNG + Markdown + CSV | 复现某个风险事件 |
| 质量摘要 | Markdown/PDF/PNG | 实验记录和论文附录 |
| 窗口样本摘要 | JSON/Markdown | 记录某个训练样本的 shape、mask 和标签 |

# 12. 流程设计

## 12.1 Run 加载流程

```text
用户选择 run_root
  -> 检查目录结构
  -> 读取 metadata.json 与 config_used.yaml
  -> 读取 scenario_table
  -> 读取质量摘要
  -> 构建 scenario index 与 risk event index
  -> 进入 Run Dashboard
```

## 12.2 动画播放流程

```text
选择 scenario_id
  -> 加载 crane_static、task_table
  -> 根据 View Mode 加载 state_true 或 state_obs
  -> 加载 geometry_table 与 edge_current
  -> Label/Debug View 时加载 edge_future_label
  -> 构建 FrameCache
  -> 用户拖动 step 或点击 Play
  -> 获取 AnimationFrame
  -> 绘制 2.5D 主图、边图层、右侧详情和时间轴
```

## 12.3 风险解释流程

```text
加载 edge_future_label
  -> 提取 risk=1 的事件
  -> 用户选择事件
  -> 定位 scenario/pair/step/horizon
  -> 加载该 pair 的 edge_current 时间序列
  -> 绘制距离曲线、阈值线、输入/未来窗口背景
  -> 绘制 t_current、t_ttc、t_min 几何快照
  -> 生成解释文本
  -> 可导出风险解释包
```

## 12.4 窗口检查流程

```text
选择 split
  -> 加载对应 windows npz
  -> 读取 sample metadata 和 feature names
  -> 用户选择 sample index
  -> 解析 node/edge/y/mask
  -> 显示 shape 与 mask
  -> 回放输入窗口和未来标签窗口
  -> 执行无泄漏检查
```

# 13. 任务设计

## 13.1 任务 A：检查一次 run 是否合格

1. 打开可视化系统并选择 run_root。
1. 进入 Run Dashboard 查看配置、场景数量、风险比例和质量 gate。
1. 筛选异常 scenario。
1. 点击异常 scenario 跳转到 Animation Player。
1. 输出 run_summary 或质量截图。

## 13.2 任务 B：播放一个 scenario 动画

1. 进入 Animation Player。
1. 选择 scenario_id 和 Input View。
1. 播放动画，观察吊臂旋转、小车 r 变化、吊钩 h 变化。
1. 打开当前边图层，查看最近边和 TTC。
1. 切换到 Debug View，在风险事件附近查看未来风险边。

## 13.3 任务 C：解释一个风险标签

1. 进入 Risk Inspector。
1. 筛选 risk_type、horizon_s 和 scene_type。
1. 选择一个风险事件。
1. 查看距离曲线、阈值线、t_current/t_ttc/t_min 快照。
1. 导出风险解释图和 Markdown 说明。

## 13.4 任务 D：检查训练窗口

1. 进入 Window Explorer。
1. 选择 train/val/test/generalization npz。
1. 筛选风险正样本窗口。
1. 检查 node_features、edge_features、y_traj、y_risk、mask。
1. 确认输入特征中没有未来标签字段。

## 13.5 任务 E：导出论文材料

1. 在 Animation Player 定位典型场景并导出当前帧。
1. 在 Risk Inspector 导出距离曲线和风险解释文本。
1. 在 Quality View 导出质量总览图。
1. 在 Export Center 打包导出。

# 14. 视觉编码规范

## 14.1 塔吊元素

| 元素 | 建议编码 |
| --- | --- |
| 基座 | 圆点 + crane_id 标签 |
| 最大作业半径 | 半透明圆 |
| 最小作业半径 | 虚线圆 |
| 吊臂 | root-tip 线段 |
| 小车/吊钩 | 吊臂上的点 + hook 高度标签 |
| pickup | 三角形 |
| dropoff | 方形 |
| 历史轨迹 | 半透明线 |
| 未来真实轨迹 | 虚线，仅 Label/Debug View |

## 14.2 风险元素

| 状态 | 建议编码 |
| --- | --- |
| 安全边 | 浅灰细线 |
| 接近风险 | 橙色线 |
| 当前风险 | 红色粗线 |
| 未来风险标签 | 红色或紫色虚线 |
| brake | 黄色 marker |
| emergency | 红色闪烁或红色 marker |
| outlier | 蓝色叉号 |
| missing | 空心点或淡化 |

## 14.3 时间窗口

| 时间段 | 建议编码 |
| --- | --- |
| 历史输入窗口 | 浅蓝背景 |
| 当前窗口末端 | 黑色竖线 |
| 未来标签窗口 | 浅红背景 |
| 首次风险点 | 红色三角 |
| 未来最小距离点 | 紫色星号 |

# 15. 无泄漏与安全设计

## 15.1 UI 层强制规则

- Input View 不读取 edge_future_label。
- Input View 不显示 risk_*、future_min_d_*、ttc_label_*。
- Label View 和 Debug View 必须显示未来信息提示。
- Window Explorer 自动检查 feature names，不允许 future/risk/label 字段进入输入。
- Data Browser 打开 edge_future_label 时显示警告。

## 15.2 推荐提示文案

```text
中文提示：
当前视图包含未来标签，仅用于验证和解释，不属于模型输入特征。

English hint:
This view contains future labels for validation only. They are not model inputs.
```

# 16. 性能设计

## 16.1 debug_small

debug_small 可以一次性加载完整 scenario 数据，主要关注播放正确性和页面功能。

## 16.2 正式数据集

- 启动时不加载所有大表，只加载 scenario_table 和 quality summary。
- 选择 scenario 后再加载该 scenario 数据。
- 优先使用 Parquet。
- 长曲线默认降采样到最多 2000 个点，导出时可使用全量数据。
- 动画播放只渲染当前帧，不一次性生成所有帧。
- 风险事件索引用 risk_ratio_by_scenario 和 edge_future_label 预处理生成。

## 16.3 分片建议

如果正式数据达到数十 GB，建议在仿真器侧额外输出按 scenario 分片的 Parquet 或 scenario_index.parquet。可视化系统先读索引，用户选择 scenario 后再读对应分片。

```text
tables_by_scenario/
  scenario_000001/
    state_true.parquet
    state_obs.parquet
    geometry_table.parquet
    edge_current.parquet
    edge_future_label.parquet
```

# 17. 开发计划

| 阶段 | 目标 | 交付物 | 验收 |
| --- | --- | --- | --- |
| V1 MVP 动画版 | 能播放 debug_small scenario | Animation Player 基础回放、图层开关、时间轴 | 能播放任意 scenario 并显示 base/root/tip/hook |
| V2 风险解释版 | 能围绕风险事件播放和解释 | Risk Inspector、风险事件列表、距离曲线、风险片段播放 | 能导出 risk=1 的解释图和说明 |
| V3 窗口检查版 | 能检查 NPZ 样本 | Window Explorer、shape/mask/feature 检查 | 能证明窗口无泄漏且标签对齐 |
| V4 质量验收版 | 能完成数据集质量验收 | Quality View、异常 scenario 列表、质量图交互 | 能判断 run 是否可用于训练 |
| V5 论文导出版 | 能导出论文/答辩材料 | Export Center、PNG/GIF/MP4/Markdown 导出 | 导出内容可直接放入论文或 PPT |

# 18. 验收标准

## 18.1 功能验收

| 编号 | 验收项 | 标准 |
| --- | --- | --- |
| F1 | Run 加载 | 能选择 run_root，缺文件时给出明确路径 |
| F2 | Dashboard | 能显示场景数量、采样率、风险比例和质量状态 |
| F3 | 动画播放 | 能播放/暂停/跳帧/拖动 step/调节速度 |
| F4 | 几何显示 | base/root/tip/hook 与 geometry_table 一致 |
| F5 | 边特征 | 能显示任意 pair 的距离、TTC 和 overlap |
| F6 | 风险事件 | 能跳转到 risk=1 事件并播放前后片段 |
| F7 | 窗口查看 | 能显示 NPZ shape、mask、y_traj、y_risk |
| F8 | 质量报告 | 能显示 quality_report、risk_ratio、feature_summary 和 plots |
| F9 | 导出 | 能导出当前帧、风险解释和质量摘要 |

## 18.2 动画验收

- 吊臂方向随 theta 变化，root-tip 线段与 geometry_table 一致。
- 小车/吊钩水平位置随 r 和 theta 变化。
- 吊钩高度 h 在详情面板和高度剖面图中正确显示。
- 作业半径圆与 crane_static 中 min_radius/max_radius 一致。
- 当前边和未来风险边显示位置、pair、risk_type 正确。
- 播放风险片段时能自动定位到事件前后窗口。

## 18.3 数据正确性验收

| 编号 | 验收项 | 标准 |
| --- | --- | --- |
| D1 | 状态一致性 | 当前 step 的状态详情与 state_true/state_obs 行一致 |
| D2 | 边一致性 | 选中 pair 的距离与 edge_current 当前行一致 |
| D3 | 标签一致性 | Risk Inspector 显示的标签与 edge_future_label 一致 |
| D4 | 时间窗口一致性 | 输入窗口为 [t-T+1,t]，未来窗口为 [t+1,t+H] |
| D5 | mask 一致性 | padding 塔吊在 node_mask/edge_mask 中为 0 |
| D6 | split 一致性 | 窗口 sample 的 scenario_id 属于对应 split |

## 18.4 无泄漏验收

- Input View 不加载 edge_future_label。
- Input View 不显示 risk、future_min_d、ttc_label。
- edge_features 中不存在 future_min_d、risk、ttc_label 等字段。
- train/val/test/generalization 按 scenario_id 互斥。
- Label View 和 Debug View 有明显未来信息提示。

## 18.5 性能验收

| 数据规模 | 验收标准 |
| --- | --- |
| debug_small | 所有页面可打开，任意 scenario 可播放，风险解释和窗口查看可用 |
| 100 scenario 正式 run | Dashboard 可快速打开，单 scenario 懒加载可播放 |
| 1000 scenario 大 run | Dashboard 只读索引不爆内存，选择 scenario 后再加载数据 |

## 18.6 UI 验收

- 用户 3 次点击内能进入 Animation Player。
- 用户能从异常 scenario 列表直接跳转到对应动画。
- 用户能从风险事件列表直接跳转到对应 step。
- 导出图包含 scenario_id、step、timestamp、view_mode、risk_type 和图例。
- 缺失文件、字段不匹配、NPZ shape 不一致时给出友好错误。

# 19. 测试方案

## 19.1 单元测试

| 测试文件 | 测试内容 |
| --- | --- |
| test_visualization_loaders.py | 读取 CSV/Parquet、缺失文件错误、metadata/config 解析 |
| test_visualization_risk_events.py | risk event 提取、risk_type 拆分、ttc_label=-1 处理 |
| test_visualization_window_view.py | NPZ shape、mask、feature_names、sample metadata 解析 |
| test_visualization_no_leakage_ui.py | Input View 不读取 edge_future_label，输入特征黑名单检查 |

## 19.2 集成测试

```text
1. 用 configs/debug_small.yaml 生成输出目录。
2. 用 loaders 打开 run_root。
3. 随机选择一个 scenario。
4. 生成一个 AnimationFrame。
5. 绘制一张 topview PNG。
6. 提取一个风险事件。
7. 生成 Risk Inspector 距离曲线。
8. 打开 train_windows.npz 并显示一个 sample。
```

## 19.3 人工验收

- 拖动时间轴时吊臂旋转是否连续。
- 小车和吊钩是否始终位于吊臂方向上。
- 任务阶段切换是否与运动趋势一致。
- brake/emergency marker 是否出现在正确时间。
- risk=1 的事件是否能在未来窗口看到距离低于阈值。
- Input View 是否没有未来标签。

# 20. 配置文件建议

建议新增 configs/visualization.yaml 控制可视化系统行为。

```text
visualization:
  default_run_root: outputs/full_version_dataset
  prefer_format: parquet
  max_points_per_curve: 2000
  playback:
    default_speed: 1.0
    available_speeds: [0.5, 1.0, 2.0, 5.0, 10.0]
    risk_clip_pre_seconds: 10.0
    risk_clip_post_seconds: 10.0
  topview:
    show_base: true
    show_jib: true
    show_hook: true
    show_radius: true
    show_task_points: true
    show_edge: true
    trail_seconds: 20.0
  risk:
    default_horizon_s: 10.0
    show_threshold_lines: true
    risk_types: [arm_arm, arm_hook_i_to_j, arm_hook_j_to_i, hook_hook]
  export:
    dpi: 300
    output_dir: exports
```

# 21. 最终交付清单

- apps/visual_dashboard.py
- src/tower_sim/visualization/loaders.py
- src/tower_sim/visualization/schemas.py
- src/tower_sim/visualization/indexing.py
- src/tower_sim/visualization/frame_cache.py
- src/tower_sim/visualization/plotting.py
- src/tower_sim/visualization/animation_player.py
- src/tower_sim/visualization/risk_events.py
- src/tower_sim/visualization/window_view.py
- src/tower_sim/visualization/quality_view.py
- src/tower_sim/visualization/export.py
- configs/visualization.yaml
- docs/visualization_design.md
- tests/test_visualization_*.py

## 21.1 启动命令

```text
streamlit run apps/visual_dashboard.py
# 或
python -m streamlit run apps/visual_dashboard.py
```

# 22. 最终验收清单

## 22.1 基础

- [ ] 可以选择 run_root。
- [ ] 可以读取 metadata.json 和 config_used.yaml。
- [ ] 可以读取 scenario_table、crane_static、state、geometry、edge。
- [ ] 可以显示 Dashboard。
- [ ] 可以进入 Animation Player。

## 22.2 动画

- [ ] 可以播放、暂停、上一帧、下一帧。
- [ ] 可以拖动 step/timestamp。
- [ ] 可以显示 base/root/tip/hook。
- [ ] 可以显示吊臂旋转、小车半径变化、吊钩高度变化。
- [ ] 可以显示作业半径、任务点、当前边、风险边。
- [ ] 可以跳转并播放风险事件前后片段。

## 22.3 风险

- [ ] 可以提取 risk event。
- [ ] 可以按 risk_type、horizon、scene_type 筛选。
- [ ] 可以显示距离曲线和阈值线。
- [ ] 可以显示输入窗口和未来窗口。
- [ ] 可以生成风险解释文本。

## 22.4 窗口

- [ ] 可以打开 train/val/test/generalization_windows.npz。
- [ ] 可以显示 node_features、edge_features、y_traj、y_risk、y_min_distance shape。
- [ ] 可以显示 node_mask、edge_mask、time_mask。
- [ ] 可以检查输入特征无 future label。

## 22.5 质量与导出

- [ ] 可以显示 quality_report.md、risk_ratio_by_scenario.csv、feature_summary.csv。
- [ ] 可以列出异常 scenario 并跳转。
- [ ] 可以导出当前帧 PNG。
- [ ] 可以导出风险片段 GIF/MP4。
- [ ] 可以导出风险解释 Markdown。

# 23. 结论

本可视化系统应被实现为“群塔时空图数据集的可视化验收平台”。它必须支持动画播放，但动画播放不是孤立目标，而是服务于数据解释和质量验收。最终系统应能够把 scenario、crane、task、state、geometry、edge_current、edge_future_label、windows 和 quality 串成一条完整证据链，证明仿真过程任务驱动、边特征具备物理先验、未来标签来自真实未来轨迹、模型输入没有未来泄漏、训练窗口和 mask 正确，数据集满足论文实验标准。
