"""可視化モジュールのテスト。

このモジュールは、heatmapとlayersモジュールの
ユニットテストを提供します。
"""

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from shadowbox.visualization import (
    create_depth_contour,
    create_depth_heatmap,
    create_depth_histogram,
    create_depth_overlay,
    create_depth_layer_comparison,
    create_labeled_image,
    create_layer_mask_preview,
    create_layer_preview,
    create_stacked_layer_view,
    show_clustering_summary,
)

# バックエンドをAggに設定（ヘッドレス環境用）
matplotlib.use("Agg")


class TestDepthHeatmap:
    """深度ヒートマップのテスト。"""

    def test_create_depth_heatmap_basic(self, sample_depth_map: np.ndarray) -> None:
        """基本的なヒートマップ作成をテスト。"""
        fig, ax = create_depth_heatmap(sample_depth_map)

        assert isinstance(fig, Figure)
        assert fig is not None

        plt_close(fig)

    def test_create_depth_heatmap_with_image(
        self,
        sample_depth_map: np.ndarray,
        sample_image_array: np.ndarray,
    ) -> None:
        """元画像付きヒートマップ作成をテスト。"""
        fig, axes = create_depth_heatmap(sample_depth_map, sample_image_array)

        assert isinstance(fig, Figure)
        # 2つのサブプロットがある
        assert len(axes) == 2

        plt_close(fig)

    def test_create_depth_heatmap_with_title(self, sample_depth_map: np.ndarray) -> None:
        """タイトル付きヒートマップをテスト。"""
        fig, ax = create_depth_heatmap(sample_depth_map, title="テスト深度マップ")

        assert isinstance(fig, Figure)

        plt_close(fig)

    def test_create_depth_heatmap_custom_cmap(self, sample_depth_map: np.ndarray) -> None:
        """カスタムカラーマップをテスト。"""
        fig, ax = create_depth_heatmap(sample_depth_map, cmap="plasma")

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestDepthOverlay:
    """深度オーバーレイのテスト。"""

    def test_create_depth_overlay_basic(
        self,
        sample_image_array: np.ndarray,
        sample_depth_map: np.ndarray,
    ) -> None:
        """基本的なオーバーレイ作成をテスト。"""
        fig, ax = create_depth_overlay(sample_image_array, sample_depth_map)

        assert isinstance(fig, Figure)

        plt_close(fig)

    def test_create_depth_overlay_custom_alpha(
        self,
        sample_image_array: np.ndarray,
        sample_depth_map: np.ndarray,
    ) -> None:
        """カスタムアルファ値をテスト。"""
        fig, ax = create_depth_overlay(sample_image_array, sample_depth_map, alpha=0.7)

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestDepthHistogram:
    """深度ヒストグラムのテスト。"""

    def test_create_depth_histogram_basic(self, sample_depth_map: np.ndarray) -> None:
        """基本的なヒストグラム作成をテスト。"""
        fig, ax = create_depth_histogram(sample_depth_map)

        assert isinstance(fig, Figure)

        plt_close(fig)

    def test_create_depth_histogram_custom_bins(self, sample_depth_map: np.ndarray) -> None:
        """カスタムビン数をテスト。"""
        fig, ax = create_depth_histogram(sample_depth_map, bins=20)

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestDepthContour:
    """深度等高線のテスト。"""

    def test_create_depth_contour_basic(self, sample_depth_map: np.ndarray) -> None:
        """基本的な等高線作成をテスト。"""
        fig, ax = create_depth_contour(sample_depth_map)

        assert isinstance(fig, Figure)

        plt_close(fig)

    def test_create_depth_contour_with_image(
        self,
        sample_depth_map: np.ndarray,
        sample_image_array: np.ndarray,
    ) -> None:
        """画像付き等高線をテスト。"""
        fig, ax = create_depth_contour(sample_depth_map, original_image=sample_image_array)

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestLayerPreview:
    """レイヤープレビューのテスト。"""

    def test_create_layer_preview_basic(
        self,
        sample_image_array: np.ndarray,
        sample_labels: np.ndarray,
        sample_centroids: np.ndarray,
    ) -> None:
        """基本的なレイヤープレビュー作成をテスト。"""
        fig, axes = create_layer_preview(
            sample_image_array,
            sample_labels,
            sample_centroids,
        )

        assert isinstance(fig, Figure)

        plt_close(fig)

    def test_create_layer_preview_single_layer(self, sample_image_array: np.ndarray) -> None:
        """単一レイヤーのプレビューをテスト。"""
        labels = np.zeros(sample_image_array.shape[:2], dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)

        fig, axes = create_layer_preview(sample_image_array, labels, centroids)

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestLayerMaskPreview:
    """レイヤーマスクプレビューのテスト。"""

    def test_create_layer_mask_preview_basic(
        self,
        sample_labels: np.ndarray,
        sample_centroids: np.ndarray,
    ) -> None:
        """基本的なマスクプレビュー作成をテスト。"""
        fig, axes = create_layer_mask_preview(sample_labels, sample_centroids)

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestLabeledImage:
    """ラベル付き画像のテスト。"""

    def test_create_labeled_image_basic(
        self,
        sample_image_array: np.ndarray,
        sample_labels: np.ndarray,
        sample_centroids: np.ndarray,
    ) -> None:
        """基本的なラベル付き画像作成をテスト。"""
        fig, ax = create_labeled_image(
            sample_image_array,
            sample_labels,
            sample_centroids,
        )

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestDepthLayerComparison:
    """深度とレイヤー比較のテスト。"""

    def test_create_depth_layer_comparison_basic(
        self,
        sample_depth_map: np.ndarray,
        sample_labels: np.ndarray,
        sample_centroids: np.ndarray,
    ) -> None:
        """基本的な比較表示作成をテスト。"""
        fig, axes = create_depth_layer_comparison(
            sample_depth_map,
            sample_labels,
            sample_centroids,
        )

        assert isinstance(fig, Figure)
        # 3つのサブプロットがある
        assert len(axes) == 3

        plt_close(fig)


class TestStackedLayerView:
    """積み重ね表示のテスト。"""

    def test_create_stacked_layer_view_basic(
        self,
        sample_image_array: np.ndarray,
        sample_labels: np.ndarray,
        sample_centroids: np.ndarray,
    ) -> None:
        """基本的な積み重ね表示作成をテスト。"""
        fig, ax = create_stacked_layer_view(
            sample_image_array,
            sample_labels,
            sample_centroids,
        )

        assert isinstance(fig, Figure)

        plt_close(fig)


class TestClusteringSummary:
    """クラスタリングサマリーのテスト。"""

    def test_show_clustering_summary_basic(
        self,
        sample_image_array: np.ndarray,
        sample_depth_map: np.ndarray,
        sample_labels: np.ndarray,
        sample_centroids: np.ndarray,
    ) -> None:
        """基本的なサマリー表示をテスト。"""
        fig, axes = show_clustering_summary(
            sample_image_array,
            sample_depth_map,
            sample_labels,
            sample_centroids,
        )

        assert isinstance(fig, Figure)
        # 複数のサブプロットがある
        assert len(axes) > 3

        plt_close(fig)


# ヘルパー関数
def plt_close(fig: Figure) -> None:
    """Figureを閉じてメモリを解放。"""
    import matplotlib.pyplot as plt

    plt.close(fig)


# フィクスチャ
@pytest.fixture
def sample_labels(sample_image_array: np.ndarray) -> np.ndarray:
    """テスト用のラベル配列を作成。画像サイズに合わせる。"""
    h, w = sample_image_array.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    labels[: h // 3, :] = 0
    labels[h // 3 : 2 * h // 3, :] = 1
    labels[2 * h // 3 :, :] = 2
    return labels


@pytest.fixture
def sample_centroids() -> np.ndarray:
    """テスト用のcentroid配列を作成。"""
    return np.array([0.2, 0.5, 0.8], dtype=np.float32)
