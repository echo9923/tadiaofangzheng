# 群塔吊时空图可视化验收系统实现说明

本项目将 `../tower_visualization_design_full_zh.md` 中的可视化方案实现为可测试的 Python 可视化层，并提供一个可选的 Streamlit 浏览器界面。

## 已实现界面

- 运行总览：`tower_sim.visualization.indexing` 汇总场景、数据划分、场景类型、塔吊数量和风险比例证据。
- 动画播放器：`tower_sim.visualization.frame_cache`、`animation_player` 和 `plotting.plot_topview_frame` 根据状态、几何、当前边和可选标签构建逐步 2.5D 帧。
- 风险解释器：`risk_events` 按风险类型拆分事件，并生成距离曲线证据和中文 Markdown 解释文本。
- 训练窗口查看器：`window_view` 打开 `*_windows.npz`，报告张量形状和 mask，筛选样本，并检查输入特征是否包含未来标签泄漏。
- 质量视图：`quality_view` 读取 `quality_report.md`、`risk_ratio_by_scenario.csv`、`feature_summary.csv` 和质量图。
- 数据浏览器：`data_browser` 支持按场景、塔吊、塔吊对、步数、时间和风险类型筛选数据表。
- 导出中心：`export` 写出当前帧 PNG、风险解释包、质量摘要和窗口样本摘要。

## 设计契约

输入视图永远不会加载或返回 `edge_future_label`。标签视图和调试视图可以展示未来标签，但界面会明确说明这些字段只用于验收，不是模型输入。输入特征黑名单会拒绝 `future_min_d`、`ttc_label`、`risk_`、`future_` 和 `label_` 等字段名，同时保留 `same_height_risk_zone` 这类当前状态特征。

实现同时支持正式 run 布局：

```text
run_root/
  tables/
  windows/
  quality/
```

以及 `outputs/debug_small` 使用的旧版平铺调试布局。
