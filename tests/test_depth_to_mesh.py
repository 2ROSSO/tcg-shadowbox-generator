"""DepthToMeshProcessorのテスト。

深度マップからメッシュへの共通変換処理のユニットテストを提供します。
"""

import numpy as np
import pytest
from PIL import Image

from shadowbox.config.settings import ClusteringSettings, RenderSettings
from shadowbox.config.template import BoundingBox
from shadowbox.core.clustering import KMeansLayerClusterer
from shadowbox.core.depth_to_mesh import (
    DepthToMeshInput,
    DepthToMeshProcessor,
    DepthToMeshResult,
)
from shadowbox.core.mesh import MeshGenerator, ShadowboxMesh


@pytest.fixture
def processor() -> DepthToMeshProcessor:
    """テスト用のDepthToMeshProcessorを作成。"""
    clusterer = KMeansLayerClusterer(ClusteringSettings())
    render_settings = RenderSettings(back_panel=False, layer_interpolation=0)
    mesh_generator = MeshGenerator(render_settings)
    return DepthToMeshProcessor(clusterer, mesh_generator)


@pytest.fixture
def sample_input() -> DepthToMeshInput:
    """テスト用の入力データを作成。"""
    height, width = 50, 60
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # 3段階の深度
    depth = np.zeros((height, width), dtype=np.float32)
    depth[:15, :] = 0.2  # 手前
    depth[15:35, :] = 0.5  # 中間
    depth[35:, :] = 0.8  # 奥

    return DepthToMeshInput(cropped_image=image, depth_map=depth)


class TestDepthToMeshProcessor:
    """DepthToMeshProcessorのテスト。"""

    def test_process_clustering(
        self, processor: DepthToMeshProcessor, sample_input: DepthToMeshInput
    ) -> None:
        """クラスタリングパスの検証。"""
        result = processor.process(sample_input, k=3, include_frame=False)

        assert isinstance(result, DepthToMeshResult)
        assert isinstance(result.mesh, ShadowboxMesh)
        assert result.optimal_k == 3
        assert len(result.centroids) == 3
        assert result.labels.shape == sample_input.depth_map.shape

    def test_process_raw_depth(
        self, processor: DepthToMeshProcessor, sample_input: DepthToMeshInput
    ) -> None:
        """生深度パスの検証。"""
        result = processor.process(
            sample_input, use_raw_depth=True, include_frame=False, depth_scale=1.0
        )

        assert isinstance(result, DepthToMeshResult)
        assert result.optimal_k == 1
        # 生深度モードではダミーのlabels/centroids
        assert np.all(result.labels == 0)
        assert len(result.centroids) == 1
        # メッシュが生成されていること
        assert result.mesh.total_vertices > 0

    def test_process_auto_k(
        self, processor: DepthToMeshProcessor, sample_input: DepthToMeshInput
    ) -> None:
        """自動k探索の検証。"""
        result = processor.process(sample_input, k=None, include_frame=False)

        assert isinstance(result, DepthToMeshResult)
        assert result.optimal_k >= 3  # ClusteringSettingsのmin_k
        assert result.optimal_k <= 10  # ClusteringSettingsのmax_k
        assert len(result.centroids) == result.optimal_k

    def test_process_with_frame(
        self, processor: DepthToMeshProcessor, sample_input: DepthToMeshInput
    ) -> None:
        """フレーム付きメッシュ生成の検証。"""
        result = processor.process(sample_input, k=3, include_frame=True)

        assert result.mesh.frame is not None

    def test_process_without_frame(
        self, processor: DepthToMeshProcessor, sample_input: DepthToMeshInput
    ) -> None:
        """フレームなしメッシュ生成の検証。"""
        result = processor.process(sample_input, k=3, include_frame=False)

        assert result.mesh.frame is None

    def test_process_card_frame(self, processor: DepthToMeshProcessor) -> None:
        """カードフレームラベル付けの検証。"""
        height, width = 100, 80
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), 0.5, dtype=np.float32)
        bbox = BoundingBox(x=10, y=20, width=60, height=60)

        input_data = DepthToMeshInput(
            cropped_image=image,
            depth_map=depth,
            original_image=image,
            bbox=bbox,
        )

        result = processor.process(
            input_data, k=3, include_frame=False, include_card_frame=True
        )

        # フレーム領域のラベルが-1になっていること
        assert np.any(result.labels == -1)
        # イラスト領域内のラベルは >= 0
        illustration_labels = result.labels[
            bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width
        ]
        assert np.all(illustration_labels >= 0)

    def test_process_card_frame_without_bbox(
        self, processor: DepthToMeshProcessor, sample_input: DepthToMeshInput
    ) -> None:
        """bbox=Noneでinclude_card_frame=Trueの場合、通常のクラスタリングになること。"""
        result = processor.process(
            sample_input, k=3, include_frame=False, include_card_frame=True
        )

        # bboxがないのでフレームラベルは付かない
        assert not np.any(result.labels == -1)


class TestDepthToMeshInput:
    """DepthToMeshInputデータクラスのテスト。"""

    def test_create_basic(self) -> None:
        """基本的な入力データ作成。"""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.zeros((10, 10), dtype=np.float32)

        input_data = DepthToMeshInput(cropped_image=image, depth_map=depth)

        assert np.array_equal(input_data.cropped_image, image)
        assert np.array_equal(input_data.depth_map, depth)
        assert input_data.original_image is None
        assert input_data.bbox is None

    def test_create_with_optional_fields(self) -> None:
        """オプションフィールド付きの入力データ作成。"""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.zeros((10, 10), dtype=np.float32)
        bbox = BoundingBox(x=1, y=2, width=5, height=5)

        input_data = DepthToMeshInput(
            cropped_image=image,
            depth_map=depth,
            original_image=image,
            bbox=bbox,
        )

        assert input_data.original_image is not None
        assert input_data.bbox == bbox


class TestDepthToMeshResult:
    """DepthToMeshResultデータクラスのテスト。"""

    def test_result_fields(self) -> None:
        """結果フィールドの検証。"""
        from unittest.mock import MagicMock

        mesh = MagicMock(spec=ShadowboxMesh)
        labels = np.zeros((10, 10), dtype=np.int32)
        centroids = np.array([0.2, 0.5, 0.8], dtype=np.float32)

        result = DepthToMeshResult(
            mesh=mesh, labels=labels, centroids=centroids, optimal_k=3
        )

        assert result.mesh == mesh
        assert result.optimal_k == 3
        assert len(result.centroids) == 3
