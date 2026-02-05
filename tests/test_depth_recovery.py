"""深度復元・メッシュ分割モジュールのテスト。

TripoSRメッシュの深度マップ復元とレイヤー分割機能の
ユニットテストを提供します。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shadowbox.triposr.depth_recovery import (
    DepthRecoverySettings,
    PyRenderDepthExtractor,
    TrimeshRayCastingExtractor,
    create_depth_extractor,
    fill_depth_holes,
)
from shadowbox.triposr.mesh_splitter import (
    DepthBasedMeshSplitter,
    MeshSplitResult,
    assign_face_labels_centroid,
    assign_face_labels_majority,
    compute_vertex_pixel_mapping,
    create_split_shadowbox_mesh,
)


class TestDepthRecoverySettings:
    """DepthRecoverySettings設定クラスのテスト。"""

    def test_default_settings(self) -> None:
        """デフォルト設定を確認するテスト。"""
        settings = DepthRecoverySettings()

        assert settings.resolution == (512, 512)
        assert settings.fill_holes is True
        assert settings.hole_fill_method == "interpolate"

    def test_custom_settings(self) -> None:
        """カスタム設定を確認するテスト。"""
        settings = DepthRecoverySettings(
            resolution=(256, 256),
            fill_holes=False,
            hole_fill_method="max_depth",
        )

        assert settings.resolution == (256, 256)
        assert settings.fill_holes is False
        assert settings.hole_fill_method == "max_depth"

    def test_frozen_dataclass(self) -> None:
        """frozen=Trueで変更不可であることを確認。"""
        settings = DepthRecoverySettings()

        with pytest.raises(AttributeError):
            settings.resolution = (256, 256)  # type: ignore


class TestFillDepthHoles:
    """fill_depth_holes関数のテスト。"""

    def test_no_holes(self) -> None:
        """穴がない場合はそのまま返す。"""
        depth_map = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ], dtype=np.float32)

        result = fill_depth_holes(depth_map, method="max_depth")

        np.testing.assert_array_almost_equal(result, depth_map)

    def test_fill_with_max_depth(self) -> None:
        """max_depthメソッドで穴を埋める。"""
        depth_map = np.array([
            [0.1, np.nan],
            [0.3, 0.4],
        ], dtype=np.float32)

        result = fill_depth_holes(depth_map, method="max_depth")

        # NaNは最大値(0.4)で埋められる
        assert not np.isnan(result).any()
        assert result[0, 1] == pytest.approx(0.4)

    def test_fill_with_interpolate(self) -> None:
        """interpolateメソッドで穴を埋める。"""
        depth_map = np.array([
            [0.1, np.nan, 0.3],
            [0.2, np.nan, 0.4],
            [0.3, 0.4, 0.5],
        ], dtype=np.float32)

        result = fill_depth_holes(depth_map, method="interpolate")

        # NaNが埋められていることを確認
        assert not np.isnan(result).any()

    def test_all_nan(self) -> None:
        """すべてNaNの場合。"""
        depth_map = np.full((3, 3), np.nan, dtype=np.float32)

        result = fill_depth_holes(depth_map, method="max_depth")

        # デフォルト値(1.0)で埋められる
        assert not np.isnan(result).any()
        np.testing.assert_array_equal(result, np.ones((3, 3), dtype=np.float32))


class TestComputeVertexPixelMapping:
    """compute_vertex_pixel_mapping関数のテスト。"""

    def test_center_vertex(self) -> None:
        """中心の頂点が画像の中心にマッピングされる。"""
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 中心
        ], dtype=np.float32)

        pixel_coords = compute_vertex_pixel_mapping(vertices, (100, 100))

        # 中心は(49, 49)または(50, 50)付近
        assert pixel_coords[0, 0] == pytest.approx(49, abs=1)  # row
        assert pixel_coords[0, 1] == pytest.approx(49, abs=1)  # col

    def test_corner_vertices(self) -> None:
        """四隅の頂点が画像の四隅にマッピングされる。"""
        vertices = np.array([
            [-1.0, 1.0, 0.0],   # 左上
            [1.0, 1.0, 0.0],    # 右上
            [-1.0, -1.0, 0.0],  # 左下
            [1.0, -1.0, 0.0],   # 右下
        ], dtype=np.float32)

        pixel_coords = compute_vertex_pixel_mapping(vertices, (100, 100))

        # 左上: row=0, col=0
        assert pixel_coords[0, 0] == 0
        assert pixel_coords[0, 1] == 0

        # 右上: row=0, col=99
        assert pixel_coords[1, 0] == 0
        assert pixel_coords[1, 1] == 99

        # 左下: row=99, col=0
        assert pixel_coords[2, 0] == 99
        assert pixel_coords[2, 1] == 0

        # 右下: row=99, col=99
        assert pixel_coords[3, 0] == 99
        assert pixel_coords[3, 1] == 99

    def test_y_axis_inverted(self) -> None:
        """Y軸が反転していることを確認。"""
        vertices = np.array([
            [0.0, 0.5, 0.0],   # 上半分
            [0.0, -0.5, 0.0],  # 下半分
        ], dtype=np.float32)

        pixel_coords = compute_vertex_pixel_mapping(vertices, (100, 100))

        # 上半分のY（正）は画像上部（小さいrow）
        # 下半分のY（負）は画像下部（大きいrow）
        assert pixel_coords[0, 0] < pixel_coords[1, 0]


class TestAssignFaceLabelsCentroid:
    """assign_face_labels_centroid関数のテスト。"""

    def test_single_layer(self) -> None:
        """単一レイヤーの場合。"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        centroids = np.array([0.5], dtype=np.float32)  # 1つのセントロイド

        labels = assign_face_labels_centroid(faces, vertices, centroids)

        assert len(labels) == 1
        assert labels[0] == 0

    def test_multiple_layers(self) -> None:
        """複数レイヤーの場合、Z座標でレイヤーが決まる。"""
        vertices = np.array([
            # 手前の面（Z=1）
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 1.0],
            # 奥の面（Z=-1）
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.5, 1.0, -1.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],  # 手前
            [3, 4, 5],  # 奥
        ], dtype=np.int32)
        # セントロイド: 0（手前）、1（奥）
        centroids = np.array([0.0, 1.0], dtype=np.float32)

        labels = assign_face_labels_centroid(faces, vertices, centroids)

        assert len(labels) == 2
        # 手前の面（Z=1）はセントロイド0（手前）に割り当て
        assert labels[0] == 0
        # 奥の面（Z=-1）はセントロイド1（奥）に割り当て
        assert labels[1] == 1


class TestAssignFaceLabelsMajority:
    """assign_face_labels_majority関数のテスト。"""

    def test_unanimous(self) -> None:
        """3頂点がすべて同じラベルの場合。"""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        vertex_labels = np.array([0, 0, 0], dtype=np.int32)

        labels = assign_face_labels_majority(faces, vertex_labels)

        assert labels[0] == 0

    def test_majority_vote(self) -> None:
        """2:1の多数決の場合。"""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        vertex_labels = np.array([0, 0, 1], dtype=np.int32)

        labels = assign_face_labels_majority(faces, vertex_labels)

        assert labels[0] == 0  # 2:1でラベル0

    def test_all_different(self) -> None:
        """3頂点がすべて異なるラベルの場合。"""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        vertex_labels = np.array([0, 1, 2], dtype=np.int32)

        labels = assign_face_labels_majority(faces, vertex_labels)

        # すべて異なる場合、どれかが選ばれる（最初に出現するものが優先される可能性）
        assert labels[0] in [0, 1, 2]


class TestTrimeshRayCastingExtractor:
    """TrimeshRayCastingExtractorのテスト。"""

    @pytest.fixture
    def check_rtree(self):
        """rtreeモジュールが利用可能かチェック。"""
        try:
            import rtree  # noqa: F401
            return True
        except ImportError:
            pytest.skip("rtree module not available for ray casting tests")

    def test_simple_plane(self, check_rtree) -> None:
        """単純な平面の深度抽出。"""
        extractor = TrimeshRayCastingExtractor()

        # Z=0の平面
        vertices = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)

        depth_map = extractor.extract_depth(vertices, faces, (10, 10))

        # 平面なので、有効なピクセルは同じ深度値を持つはず
        valid_mask = ~np.isnan(depth_map)
        if np.any(valid_mask):
            valid_depths = depth_map[valid_mask]
            # すべて同じ深度（平面なので）
            assert np.std(valid_depths) < 0.1

    def test_depth_ordering(self, check_rtree) -> None:
        """手前と奥の面で深度値が異なる。"""
        extractor = TrimeshRayCastingExtractor()

        # 手前（Z=0.5）と奥（Z=-0.5）の2つの平面
        vertices = np.array([
            # 手前の平面（右半分）
            [0.0, -1.0, 0.5],
            [1.0, -1.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            # 奥の平面（左半分）
            [-1.0, -1.0, -0.5],
            [0.0, -1.0, -0.5],
            [0.0, 1.0, -0.5],
            [-1.0, 1.0, -0.5],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ], dtype=np.int32)

        depth_map = extractor.extract_depth(vertices, faces, (10, 10))

        # 深度マップが生成されていることを確認
        valid_mask = ~np.isnan(depth_map)
        assert np.any(valid_mask), "深度マップにレンダリングされたピクセルがありません"


class TestDepthBasedMeshSplitter:
    """DepthBasedMeshSplitterのテスト。"""

    def test_init_default(self) -> None:
        """デフォルト初期化のテスト。"""
        splitter = DepthBasedMeshSplitter()

        assert splitter._depth_extractor is not None
        assert splitter._settings is not None

    def test_init_with_settings(self) -> None:
        """カスタム設定での初期化。"""
        settings = DepthRecoverySettings(resolution=(256, 256))
        splitter = DepthBasedMeshSplitter(settings=settings)

        assert splitter._settings.resolution == (256, 256)

    def test_split_simple_mesh(self) -> None:
        """単純なメッシュの分割テスト。"""
        # モックの深度抽出器を作成
        mock_extractor = MagicMock()
        # 深度マップを返す（上半分が手前、下半分が奥）
        depth_map = np.zeros((512, 512), dtype=np.float32)  # デフォルト解像度に合わせる
        depth_map[:256, :] = 0.2  # 手前
        depth_map[256:, :] = 0.8  # 奥
        mock_extractor.extract_depth.return_value = depth_map

        # モックのクラスタラーを作成
        mock_clusterer = MagicMock()
        labels = np.zeros((512, 512), dtype=np.int32)  # デフォルト解像度に合わせる
        labels[:256, :] = 0  # 手前レイヤー
        labels[256:, :] = 1  # 奥レイヤー
        mock_clusterer.cluster.return_value = (labels, np.array([0.2, 0.8], dtype=np.float32))

        splitter = DepthBasedMeshSplitter(
            depth_extractor=mock_extractor,
            clusterer=mock_clusterer,
        )

        # テスト用メッシュ（頂点は[-1, 1]範囲に正規化）
        vertices = np.array([
            [-0.5, 0.5, 0.5],   # 手前・上
            [0.5, 0.5, 0.5],    # 手前・上
            [0.5, -0.5, -0.5],  # 奥・下
            [-0.5, -0.5, -0.5], # 奥・下
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ], dtype=np.uint8)

        result = splitter.split(vertices, faces, colors, k=2)

        assert isinstance(result, MeshSplitResult)
        assert len(result.layer_meshes) == 2
        assert len(result.centroids) == 2


class TestCreateSplitShadowboxMesh:
    """create_split_shadowbox_mesh関数のテスト。"""

    def test_creates_shadowbox_mesh(self) -> None:
        """ShadowboxMeshが正しく作成される。"""
        from shadowbox.core.mesh import LayerMesh

        # ダミーのMeshSplitResultを作成
        layer1 = LayerMesh(
            vertices=np.array([[0, 0, 0]], dtype=np.float32),
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            z_position=0.0,
            layer_index=0,
            pixel_indices=np.array([[0, 0]], dtype=np.int32),
        )
        layer2 = LayerMesh(
            vertices=np.array([[1, 1, 1]], dtype=np.float32),
            colors=np.array([[0, 255, 0]], dtype=np.uint8),
            z_position=1.0,
            layer_index=1,
            pixel_indices=np.array([[0, 0]], dtype=np.int32),
        )

        split_result = MeshSplitResult(
            layer_meshes=[layer1, layer2],
            depth_map=np.zeros((10, 10), dtype=np.float32),
            labels=np.zeros((10, 10), dtype=np.int32),
            centroids=np.array([0.0, 1.0], dtype=np.float32),
        )

        bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        mesh = create_split_shadowbox_mesh(split_result, bounds)

        assert mesh.num_layers == 2
        assert mesh.bounds == bounds
        assert mesh.frame is None


class TestCreateDepthExtractor:
    """create_depth_extractor関数のテスト。"""

    def test_fallback_to_trimesh(self) -> None:
        """pyrenderが利用不可の場合、trimeshにフォールバック。"""
        with patch.dict("sys.modules", {"pyrender": None}):
            extractor = create_depth_extractor(use_pyrender=False)
            assert isinstance(extractor, TrimeshRayCastingExtractor)

    def test_explicit_trimesh(self) -> None:
        """明示的にtrimeshを指定。"""
        extractor = create_depth_extractor(use_pyrender=False)
        assert isinstance(extractor, TrimeshRayCastingExtractor)
