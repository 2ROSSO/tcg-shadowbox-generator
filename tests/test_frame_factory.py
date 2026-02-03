"""フレームファクトリモジュールのテスト。

このモジュールは、frame_factoryモジュールの
ユニットテストを提供します。
"""

import numpy as np
import pytest

from shadowbox.core.frame_factory import (
    FrameConfig,
    calculate_bounds,
    create_frame,
    create_plane_frame,
    create_walled_frame,
)
from shadowbox.core.mesh import FrameMesh, LayerMesh


class TestFrameConfig:
    """FrameConfigデータクラスのテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値のテスト。"""
        config = FrameConfig()

        assert config.z_front == 0.0
        assert config.z_back is None
        assert config.margin == 0.05
        assert config.frame_color == (30, 30, 30)

    def test_custom_values(self) -> None:
        """カスタム値のテスト。"""
        config = FrameConfig(
            z_front=0.5,
            z_back=-1.0,
            margin=0.1,
            frame_color=(50, 50, 50),
        )

        assert config.z_front == 0.5
        assert config.z_back == -1.0
        assert config.margin == 0.1
        assert config.frame_color == (50, 50, 50)

    def test_frozen(self) -> None:
        """frozenデータクラスであることを確認。"""
        config = FrameConfig()

        with pytest.raises(AttributeError):
            config.z_front = 1.0  # type: ignore


class TestCreateFrame:
    """create_frame関数のテスト。"""

    def test_plane_frame_when_no_z_back(self) -> None:
        """z_backがNoneの場合、平面フレームを生成。"""
        config = FrameConfig(z_front=0.0, z_back=None)
        frame = create_frame(config)

        assert isinstance(frame, FrameMesh)
        assert frame.has_walls is False
        assert len(frame.vertices) == 8

    def test_walled_frame_when_z_back_specified(self) -> None:
        """z_backが指定されている場合、壁付きフレームを生成。"""
        config = FrameConfig(z_front=0.0, z_back=-0.5)
        frame = create_frame(config)

        assert isinstance(frame, FrameMesh)
        assert frame.has_walls is True
        assert len(frame.vertices) == 12


class TestCreatePlaneFrame:
    """create_plane_frame関数のテスト。"""

    def test_vertex_count(self) -> None:
        """頂点数が8であることを確認。"""
        config = FrameConfig()
        frame = create_plane_frame(config)

        assert len(frame.vertices) == 8

    def test_face_count(self) -> None:
        """面数が8であることを確認。"""
        config = FrameConfig()
        frame = create_plane_frame(config)

        assert len(frame.faces) == 8

    def test_z_positions(self) -> None:
        """すべての頂点が同じZ座標を持つことを確認。"""
        z_front = 0.5
        config = FrameConfig(z_front=z_front)
        frame = create_plane_frame(config)

        assert np.allclose(frame.vertices[:, 2], z_front)

    def test_z_position_attribute(self) -> None:
        """z_position属性が正しく設定されていることを確認。"""
        z_front = 0.5
        config = FrameConfig(z_front=z_front)
        frame = create_plane_frame(config)

        assert frame.z_position == z_front

    def test_has_walls_false(self) -> None:
        """has_wallsがFalseであることを確認。"""
        config = FrameConfig()
        frame = create_plane_frame(config)

        assert frame.has_walls is False

    def test_z_back_is_none(self) -> None:
        """z_backがNoneであることを確認。"""
        config = FrameConfig()
        frame = create_plane_frame(config)

        assert frame.z_back is None

    def test_outer_vertices_have_margin(self) -> None:
        """外側頂点がマージンを持つことを確認。"""
        margin = 0.1
        config = FrameConfig(margin=margin)
        frame = create_plane_frame(config)

        outer = 1.0 + margin
        # 外側頂点（0-3）のx, y座標がouter値を持つ
        assert abs(frame.vertices[0, 0]) == pytest.approx(outer)
        assert abs(frame.vertices[0, 1]) == pytest.approx(outer)

    def test_inner_vertices_at_unit_boundary(self) -> None:
        """内側頂点が単位境界にあることを確認。"""
        config = FrameConfig()
        frame = create_plane_frame(config)

        # 内側頂点（4-7）のx, y座標が1.0
        assert abs(frame.vertices[4, 0]) == pytest.approx(1.0)
        assert abs(frame.vertices[4, 1]) == pytest.approx(1.0)

    def test_color(self) -> None:
        """色が正しく設定されていることを確認。"""
        color = (100, 150, 200)
        config = FrameConfig(frame_color=color)
        frame = create_plane_frame(config)

        assert np.array_equal(frame.color, np.array(color, dtype=np.uint8))


class TestCreateWalledFrame:
    """create_walled_frame関数のテスト。"""

    def test_vertex_count(self) -> None:
        """頂点数が12であることを確認。"""
        config = FrameConfig(z_back=-0.5)
        frame = create_walled_frame(config)

        assert len(frame.vertices) == 12

    def test_face_count(self) -> None:
        """面数が16であることを確認。"""
        config = FrameConfig(z_back=-0.5)
        frame = create_walled_frame(config)

        assert len(frame.faces) == 16

    def test_z_positions_front_and_back(self) -> None:
        """前面と背面のZ座標が正しいことを確認。"""
        z_front = 0.5
        z_back = -1.0
        config = FrameConfig(z_front=z_front, z_back=z_back)
        frame = create_walled_frame(config)

        # 前面頂点（0-7）
        front_vertices = frame.vertices[:8]
        assert np.allclose(front_vertices[:, 2], z_front)

        # 背面頂点（8-11）
        back_vertices = frame.vertices[8:12]
        assert np.allclose(back_vertices[:, 2], z_back)

    def test_has_walls_flag(self) -> None:
        """has_wallsがTrueであることを確認。"""
        config = FrameConfig(z_back=-0.5)
        frame = create_walled_frame(config)

        assert frame.has_walls is True

    def test_z_back_attribute(self) -> None:
        """z_back属性が正しく設定されていることを確認。"""
        z_back = -0.8
        config = FrameConfig(z_back=z_back)
        frame = create_walled_frame(config)

        assert frame.z_back == z_back

    def test_z_position_attribute(self) -> None:
        """z_position属性が正しく設定されていることを確認。"""
        z_front = 0.3
        config = FrameConfig(z_front=z_front, z_back=-0.5)
        frame = create_walled_frame(config)

        assert frame.z_position == z_front

    def test_raises_error_when_z_back_none(self) -> None:
        """z_backがNoneの場合にValueErrorを発生させることを確認。"""
        config = FrameConfig(z_back=None)

        with pytest.raises(ValueError, match="z_back must be specified"):
            create_walled_frame(config)

    def test_color(self) -> None:
        """色が正しく設定されていることを確認。"""
        color = (80, 80, 80)
        config = FrameConfig(z_back=-0.5, frame_color=color)
        frame = create_walled_frame(config)

        assert np.array_equal(frame.color, np.array(color, dtype=np.uint8))


class TestCalculateBounds:
    """calculate_bounds関数のテスト。"""

    def _create_layer(
        self,
        vertices: np.ndarray,
        z_position: float = 0.0,
        layer_index: int = 0,
    ) -> LayerMesh:
        """テスト用のLayerMeshを作成。"""
        num_vertices = len(vertices)
        return LayerMesh(
            vertices=vertices.astype(np.float32),
            colors=np.zeros((num_vertices, 3), dtype=np.uint8),
            z_position=z_position,
            layer_index=layer_index,
            pixel_indices=np.zeros((num_vertices, 2), dtype=np.int32),
        )

    def test_with_layers_only(self) -> None:
        """レイヤーのみでバウンズ計算。"""
        vertices = np.array([
            [-0.5, -0.5, -0.3],
            [0.5, 0.5, 0.2],
        ])
        layer = self._create_layer(vertices)

        bounds = calculate_bounds([layer], frame=None)

        assert bounds == pytest.approx((-0.5, 0.5, -0.5, 0.5, -0.3, 0.2))

    def test_with_frame(self) -> None:
        """レイヤーとフレームでバウンズ計算。"""
        # 小さいレイヤー（フレーム内に収まる）
        layer_vertices = np.array([
            [-0.3, -0.3, -0.2],
            [0.3, 0.3, -0.1],
        ])
        layer = self._create_layer(layer_vertices)

        # 大きいフレーム
        config = FrameConfig(z_front=0.0, z_back=-0.5)
        frame = create_walled_frame(config)

        bounds = calculate_bounds([layer], frame)

        # フレームの方が大きいのでフレームのバウンズになる
        assert bounds[0] == pytest.approx(-1.05)  # min_x (outer)
        assert bounds[1] == pytest.approx(1.05)   # max_x (outer)
        assert bounds[4] == pytest.approx(-0.5)   # min_z (z_back)
        assert bounds[5] == pytest.approx(0.0)    # max_z (z_front)

    def test_empty_layers(self) -> None:
        """空のレイヤーリストでデフォルトバウンズを返す。"""
        bounds = calculate_bounds([], frame=None)

        assert bounds == (-1.0, 1.0, -1.0, 1.0, -1.0, 0.0)

    def test_multiple_layers(self) -> None:
        """複数レイヤーでバウンズ計算。"""
        layer1_vertices = np.array([[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0]])
        layer2_vertices = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        layer1 = self._create_layer(layer1_vertices, layer_index=0)
        layer2 = self._create_layer(layer2_vertices, layer_index=1)

        bounds = calculate_bounds([layer1, layer2], frame=None)

        assert bounds == pytest.approx((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))

    def test_empty_layer_ignored(self) -> None:
        """空頂点のレイヤーは無視される。"""
        empty_layer = self._create_layer(np.array([]).reshape(0, 3))
        filled_layer = self._create_layer(
            np.array([[-0.3, -0.3, -0.1], [0.3, 0.3, 0.1]])
        )

        bounds = calculate_bounds([empty_layer, filled_layer], frame=None)

        assert bounds == pytest.approx((-0.3, 0.3, -0.3, 0.3, -0.1, 0.1))

    def test_frame_only(self) -> None:
        """フレームのみでバウンズ計算（空レイヤーリスト）。"""
        config = FrameConfig(z_front=0.0, z_back=-0.5)
        frame = create_walled_frame(config)

        bounds = calculate_bounds([], frame)

        assert bounds[0] == pytest.approx(-1.05)  # min_x
        assert bounds[1] == pytest.approx(1.05)   # max_x
        assert bounds[4] == pytest.approx(-0.5)   # min_z
        assert bounds[5] == pytest.approx(0.0)    # max_z
