from __future__ import annotations

from typing import Any

import pandas as pd

from tower_sim.visualization.schemas import ViewMode


APP_TITLE = "群塔吊时空图可视化验收系统"
UNTRANSLATED_FIELD_NOTE = "未翻译字段为原始数据字段或模型特征名。"

PAGE_LABELS = {
    "Run Dashboard": "运行总览",
    "Animation Player": "动画播放器",
    "Risk Inspector": "风险解释器",
    "Window Explorer": "训练窗口查看器",
    "Quality View": "质量视图",
    "Data Browser": "数据浏览器",
}

VIEW_MODE_LABELS = {
    ViewMode.INPUT.value: "输入视图",
    ViewMode.LABEL.value: "标签视图",
    ViewMode.DEBUG.value: "调试视图",
}

SPLIT_LABELS = {
    "train": "训练集",
    "val": "验证集",
    "test": "测试集",
    "generalization": "泛化集",
}

RISK_TYPE_LABELS = {
    "arm_arm": "臂-臂",
    "arm_hook_i_to_j": "臂-钩 i->j",
    "arm_hook_j_to_i": "臂-钩 j->i",
    "hook_hook": "钩-钩",
}

RISK_COLUMN_LABELS = {
    "risk_arm_arm": "臂-臂风险",
    "risk_arm_hook_i_to_j": "臂-钩 i->j 风险",
    "risk_arm_hook_j_to_i": "臂-钩 j->i 风险",
    "risk_hook_hook": "钩-钩风险",
}

FIELD_LABELS = {
    "scenario_id": "场景ID",
    "scenario_uid": "场景UID",
    "scenario_index": "场景索引",
    "scene_type": "场景类型",
    "num_cranes": "塔吊数量",
    "duration_s": "持续时间/s",
    "dt": "时间步长/s",
    "site_length_m": "场地长度/m",
    "site_width_m": "场地宽度/m",
    "seed": "随机种子",
    "split": "数据划分",
    "layout_relabel_reason": "布局重标记原因",
    "risk_any_ratio": "总体风险比例",
    "risk_arm_arm_ratio": "臂-臂风险比例",
    "risk_arm_hook_ratio": "臂-钩风险比例",
    "risk_hook_hook_ratio": "钩-钩风险比例",
    "anomaly_reason": "异常原因",
    "source": "来源",
    "feature": "特征",
    "count": "计数",
    "mean": "均值",
    "std": "标准差",
    "min": "最小值",
    "25%": "25%分位",
    "50%": "中位数",
    "75%": "75%分位",
    "max": "最大值",
    "crane_id": "塔吊ID",
    "crane_uid": "塔吊UID",
    "crane_index": "塔吊索引",
    "base_x": "基座X/m",
    "base_y": "基座Y/m",
    "base_z": "基座Z/m",
    "tower_height": "塔高/m",
    "jib_length": "吊臂长度/m",
    "min_radius": "最小作业半径/m",
    "max_radius": "最大作业半径/m",
    "safety_radius_arm": "吊臂安全半径/m",
    "safety_radius_hook": "吊钩安全半径/m",
    "priority": "优先级",
    "task_id": "任务ID",
    "task_uid": "任务UID",
    "task_index": "任务索引",
    "start_time": "开始时间/s",
    "pickup_theta": "取货角度/rad",
    "pickup_r": "取货半径/m",
    "pickup_h": "取货高度/m",
    "dropoff_theta": "卸货角度/rad",
    "dropoff_r": "卸货半径/m",
    "dropoff_h": "卸货高度/m",
    "transport_h": "运输高度/m",
    "load_weight": "载荷重量/kg",
    "timestamp": "时间/s",
    "step": "步数",
    "theta": "角度/rad",
    "r": "半径/m",
    "h": "吊钩高度/m",
    "theta_dot": "角速度/rad/s",
    "r_dot": "径向速度/m/s",
    "h_dot": "升降速度/m/s",
    "theta_ddot": "角加速度/rad/s^2",
    "r_ddot": "径向加速度/m/s^2",
    "h_ddot": "升降加速度/m/s^2",
    "command_theta": "角速度指令",
    "command_r": "径向速度指令",
    "command_h": "升降速度指令",
    "brake_flag": "制动标记",
    "emergency_flag": "急停标记",
    "task_stage": "任务阶段",
    "theta_missing": "角度缺失",
    "r_missing": "半径缺失",
    "h_missing": "高度缺失",
    "obs_delay_steps": "观测延迟步数",
    "is_outlier": "异常观测",
    "root_x": "臂根X/m",
    "root_y": "臂根Y/m",
    "root_z": "臂根Z/m",
    "tip_x": "臂端X/m",
    "tip_y": "臂端Y/m",
    "tip_z": "臂端Z/m",
    "hook_x": "吊钩X/m",
    "hook_y": "吊钩Y/m",
    "hook_z": "吊钩Z/m",
    "crane_i": "塔吊i",
    "crane_j": "塔吊j",
    "crane_i_uid": "塔吊i UID",
    "crane_j_uid": "塔吊j UID",
    "crane_i_index": "塔吊i索引",
    "crane_j_index": "塔吊j索引",
    "d_arm_arm": "臂-臂距离/m",
    "d_arm_hook_i_to_j": "臂-钩 i->j 距离/m",
    "d_arm_hook_j_to_i": "臂-钩 j->i 距离/m",
    "d_hook_hook": "钩-钩距离/m",
    "delta_theta": "角度差/rad",
    "delta_theta_dot": "角速度差/rad/s",
    "delta_r": "半径差/m",
    "delta_h": "高度差/m",
    "delta_tower_height": "塔高差/m",
    "base_distance": "基座距离/m",
    "overlap_ratio": "作业半径重叠比例",
    "relative_approach_speed": "相对接近速度/m/s",
    "relative_approach_speed_arm_arm": "臂-臂接近速度/m/s",
    "relative_approach_speed_arm_hook": "臂-钩接近速度/m/s",
    "relative_approach_speed_hook_hook": "钩-钩接近速度/m/s",
    "ttc_est_arm_arm": "臂-臂当前TTC估计/s",
    "ttc_est_arm_hook": "臂-钩当前TTC估计/s",
    "ttc_est_hook_hook": "钩-钩当前TTC估计/s",
    "same_height_risk_zone": "同高度风险区",
    "horizon_s": "预测窗口/s",
    "future_min_d_arm_arm": "未来最小臂-臂距离/m",
    "future_min_d_arm_hook_i_to_j": "未来最小臂-钩 i->j 距离/m",
    "future_min_d_arm_hook_j_to_i": "未来最小臂-钩 j->i 距离/m",
    "future_min_d_hook_hook": "未来最小钩-钩距离/m",
    "ttc_label_arm_arm": "臂-臂标签TTC/s",
    "ttc_label_arm_hook": "臂-钩标签TTC/s",
    "ttc_label_hook_hook": "钩-钩标签TTC/s",
    **RISK_COLUMN_LABELS,
}

METRIC_LABELS = {
    "run_root": "运行目录",
    "num_scenarios": "场景数",
    "metadata": "元数据",
    "split_counts": "数据划分数量",
    "scene_type_counts": "场景类型数量",
    "num_cranes_distribution": "塔吊数量分布",
    "risk_any_ratio": "总体风险比例",
    "risk_any_ratio_min": "最低总体风险比例",
    "risk_any_ratio_max": "最高总体风险比例",
    "risk_any_ratio_mean": "平均总体风险比例",
    "feature_summary_rows": "特征摘要行数",
    "quality_plots": "质量图数量",
    "node_features": "节点特征",
    "edge_features": "边特征",
    "y_traj": "未来轨迹标签",
    "y_risk": "未来风险标签",
    "y_min_distance": "未来最小距离标签",
    "node_mask": "节点掩码",
    "edge_mask": "边掩码",
    "time_mask": "时间掩码",
    "node_feature_names": "节点特征名",
    "edge_feature_names": "边特征名",
    "y_traj_feature_names": "未来轨迹标签名",
    "y_risk_feature_names": "未来风险标签名",
    "y_min_distance_feature_names": "未来最小距离标签名",
    "path": "文件路径",
    "num_samples": "样本数",
    "shapes": "张量形状",
    "node_feature_names": "节点特征名",
    "edge_feature_names": "边特征名",
    "leakage_offenders": "泄漏字段",
    "scenario_count": "场景数量",
    "risk_positive_samples": "风险正样本数",
    "sample_index": "样本索引",
    "window_start_step": "窗口起始步数",
    "risk_positive_count": "风险正标签数量",
    "node_mask_true": "有效节点数",
    "edge_mask_true": "有效边数",
    "time_mask_true": "有效时间步数",
}

ANOMALY_REASON_LABELS = {
    "missing risk ratio": "缺少风险比例",
    "risk ratio too low": "风险比例过低",
    "risk ratio too high": "风险比例过高",
}


def page_label(page_key: str) -> str:
    return PAGE_LABELS.get(page_key, page_key)


def view_mode_label(value: str | ViewMode) -> str:
    mode = ViewMode.parse(value)
    return VIEW_MODE_LABELS.get(mode.value, mode.value)


def split_label(value: Any) -> str:
    return SPLIT_LABELS.get(str(value), str(value))


def risk_type_label(value: str) -> str:
    return RISK_TYPE_LABELS.get(str(value), str(value))


def field_label(value: str) -> str:
    return FIELD_LABELS.get(str(value), METRIC_LABELS.get(str(value), str(value)))


def rename_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: field_label(str(column)) for column in df.columns})


def localize_value(key: str, value: Any) -> Any:
    if key == "split":
        return split_label(value)
    if key == "risk_type":
        return risk_type_label(str(value))
    if key == "anomaly_reason":
        return ANOMALY_REASON_LABELS.get(str(value), value)
    if isinstance(value, dict):
        return {field_label(str(k)): localize_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [localize_value(key, item) for item in value]
    return value


def localize_dict_keys(data: dict[str, Any]) -> dict[str, Any]:
    return {field_label(str(key)): localize_value(str(key), value) for key, value in data.items()}


def localize_records_frame(df: pd.DataFrame) -> pd.DataFrame:
    display = rename_columns_for_display(df)
    for column in display.columns:
        if column == field_label("split"):
            display[column] = display[column].map(split_label)
        elif column == field_label("risk_type"):
            display[column] = display[column].map(risk_type_label)
        elif column == field_label("anomaly_reason"):
            display[column] = display[column].map(lambda value: ANOMALY_REASON_LABELS.get(str(value), value))
    return display


def tensor_shape_summary(shapes: dict[str, tuple[int, ...]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"张量": field_label(name), "形状": " x ".join(str(part) for part in shape)} for name, shape in shapes.items()]
    )
