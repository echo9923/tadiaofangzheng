from pathlib import Path
import uuid

from tower_sim.visualization.animation_player import assert_frame_respects_view_mode, get_animation_frame, view_mode_warning
from tower_sim.visualization.data_browser import browse_table
from tower_sim.visualization.export import export_frame_png, export_quality_summary_markdown, export_window_sample_summary
from tower_sim.visualization.i18n_zh import rename_columns_for_display, risk_type_label
from tower_sim.visualization.loaders import RunDataRepository
from tower_sim.visualization.quality_view import load_quality_view
from tower_sim.visualization.schemas import ViewMode, assert_input_columns_safe
from tower_sim.visualization.window_view import get_window_sample, load_window_bundle


def _repo() -> RunDataRepository:
    return RunDataRepository(Path(__file__).resolve().parents[1] / "outputs" / "debug_small", prefer_format="csv")


def test_input_view_contract_and_warning_text() -> None:
    repo = _repo()
    frame = get_animation_frame(repo, 0, step=0, view_mode=ViewMode.INPUT)

    assert_frame_respects_view_mode(frame)
    assert "不会加载或显示任何未来标签" in view_mode_warning(ViewMode.INPUT)
    assert "不是模型输入" in view_mode_warning(ViewMode.LABEL)


def test_input_feature_blacklist_rejects_label_like_names() -> None:
    for columns in [
        ["risk_arm_arm"],
        ["future_min_d_arm_arm"],
        ["ttc_label_arm_arm"],
        ["label_distance"],
        ["future_window_flag"],
    ]:
        try:
            assert_input_columns_safe(columns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {columns} to be rejected")

    assert_input_columns_safe(["d_arm_arm", "ttc_est_arm_arm", "same_height_risk_zone"])


def test_data_browser_warns_for_future_label_table_and_filters() -> None:
    result = browse_table(_repo(), "edge_future_label", scenario_id=0, risk_type="arm_arm")

    assert result.warning is not None
    assert "未来标签" in result.warning
    assert set(result.data["scenario_id"].unique()).issubset({0})
    if not result.data.empty:
        assert (result.data["risk_arm_arm"] > 0).all()


def test_chinese_display_headers_do_not_mutate_source_frame() -> None:
    result = browse_table(_repo(), "edge_future_label", scenario_id=0)
    original_columns = list(result.data.columns)

    display = rename_columns_for_display(result.data)

    assert list(result.data.columns) == original_columns
    assert "场景ID" in display.columns
    assert "步数" in display.columns
    assert "未来最小臂-臂距离/m" in display.columns
    assert "scenario_id" not in display.columns
    assert risk_type_label("arm_arm") == "臂-臂"


def test_quality_view_and_export_helpers() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "test_artifacts" / f"visual_export_{uuid.uuid4().hex}"
    repo = _repo()
    quality = load_quality_view(repo, low=0.0, high=1.0)

    assert "num_scenarios" in quality.dashboard_metrics
    assert not quality.report_markdown == ""
    quality_summary = export_quality_summary_markdown(quality.dashboard_metrics, output_dir / "quality.md")
    assert quality_summary.exists()
    assert "群塔吊可视化质量摘要" in quality_summary.read_text(encoding="utf-8")

    frame = get_animation_frame(repo, 0, step=0, view_mode=ViewMode.DEBUG)
    frame_path = export_frame_png(frame, output_dir / "frame.png", dpi=120)
    assert frame_path.exists()
    assert frame_path.stat().st_size > 0

    bundle = load_window_bundle(repo.windows_dir / "train_windows.npz", split="train")
    sample = get_window_sample(bundle, 0)
    sample_path = export_window_sample_summary(sample, output_dir / "sample.json")
    sample_text = sample_path.read_text(encoding="utf-8")
    assert "张量形状" in sample_text
    assert "节点特征" in sample_text
