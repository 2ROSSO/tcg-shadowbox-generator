"""back_panel_factoryモジュールのテスト。

背面パネル生成機能のユニットテストを提供します。
"""

import numpy as np
import pytest

from shadowbox.core.back_panel_factory import create_back_panel


class TestCreateBackPanel:
    """create_back_panel関数のテスト。"""

    def test_vertex_count(self) -> None:
        """頂点数が画像ピクセル数と一致することを確認。"""
        h, w = 10, 20
        image = np.zeros((h, w, 3), dtype=np.uint8)
        z = -0.5

        panel = create_back_panel(image, z)

        assert len(panel.vertices) == h * w
        assert len(panel.colors) == h * w
        assert len(panel.pixel_indices) == h * w

    def test_z_position(self) -> None:
        """全頂点が指定Z座標にあることを確認。"""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        z = -1.5

        panel = create_back_panel(image, z)

        # 全頂点のZ座標が指定値であること
        assert np.allclose(panel.vertices[:, 2], z)
        assert panel.z_position == z

    def test_colors_match_image(self) -> None:
        """頂点色が元画像のピクセル色と一致することを確認。"""
        # 各ピクセルに異なる色を設定
        image = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ], dtype=np.uint8)
        z = 0.0

        panel = create_back_panel(image, z)

        # 色がフラット化された順序と一致
        expected_colors = image.reshape(-1, 3)
        assert np.array_equal(panel.colors, expected_colors)

    def test_normalized_coordinates(self) -> None:
        """座標が[-1, 1]の範囲に正規化されていることを確認。"""
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        z = 0.0

        panel = create_back_panel(image, z)

        # X座標: [-1, 1]
        assert panel.vertices[:, 0].min() >= -1.0
        assert panel.vertices[:, 0].max() <= 1.0

        # Y座標: [-1, 1]
        assert panel.vertices[:, 1].min() >= -1.0
        assert panel.vertices[:, 1].max() <= 1.0

        # 端の値が正確に-1と1であること
        assert np.isclose(panel.vertices[:, 0].min(), -1.0)
        assert np.isclose(panel.vertices[:, 0].max(), 1.0)
        assert np.isclose(panel.vertices[:, 1].min(), -1.0)
        assert np.isclose(panel.vertices[:, 1].max(), 1.0)

    def test_layer_index(self) -> None:
        """レイヤーインデックスが正しく設定されることを確認。"""
        image = np.zeros((5, 5, 3), dtype=np.uint8)

        panel = create_back_panel(image, z=0.0, layer_index=3)

        assert panel.layer_index == 3

    def test_default_layer_index(self) -> None:
        """デフォルトのレイヤーインデックスが0であることを確認。"""
        image = np.zeros((5, 5, 3), dtype=np.uint8)

        panel = create_back_panel(image, z=0.0)

        assert panel.layer_index == 0

    def test_pixel_indices(self) -> None:
        """ピクセルインデックスが正しく設定されることを確認。"""
        h, w = 3, 4
        image = np.zeros((h, w, 3), dtype=np.uint8)

        panel = create_back_panel(image, z=0.0)

        # ピクセルインデックスの形状
        assert panel.pixel_indices.shape == (h * w, 2)

        # 各インデックスが有効な範囲内
        assert panel.pixel_indices[:, 0].min() >= 0
        assert panel.pixel_indices[:, 0].max() < h
        assert panel.pixel_indices[:, 1].min() >= 0
        assert panel.pixel_indices[:, 1].max() < w

    def test_single_pixel_image(self) -> None:
        """1x1画像でも正しく動作することを確認。"""
        image = np.array([[[128, 64, 32]]], dtype=np.uint8)
        z = -0.5

        panel = create_back_panel(image, z)

        assert len(panel.vertices) == 1
        assert np.array_equal(panel.colors[0], [128, 64, 32])
        # 1x1の場合、座標は(0, 0)に正規化される（ゼロ除算回避）
        assert panel.vertices[0, 0] == 0.0
        assert panel.vertices[0, 1] == 0.0
        assert panel.vertices[0, 2] == z

    def test_dtype(self) -> None:
        """出力のデータ型が正しいことを確認。"""
        image = np.zeros((5, 5, 3), dtype=np.uint8)

        panel = create_back_panel(image, z=0.0)

        assert panel.vertices.dtype == np.float32
        assert panel.colors.dtype == np.uint8
        assert panel.pixel_indices.dtype == np.int32
