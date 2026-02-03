"""TripoSRパイプラインモジュール。

TripoSRを使用した3Dメッシュ生成パイプラインを提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from shadowbox.config.settings import RenderSettings
from shadowbox.config.template import BoundingBox
from shadowbox.core.frame_factory import FrameConfig, calculate_bounds, create_frame
from shadowbox.core.mesh import ShadowboxMesh
from shadowbox.triposr.generator import TripoSRGenerator
from shadowbox.triposr.settings import TripoSRSettings
from shadowbox.utils.image import crop_image, image_to_array, load_image


@dataclass
class TripoSRPipelineResult:
    """TripoSRパイプラインの実行結果。

    Attributes:
        original_image: 元の入力画像（NumPy配列）。
        mesh: 生成された3Dメッシュ。
        bbox: 使用されたバウンディングボックス（クロップした場合）。
    """

    original_image: NDArray[np.uint8]
    mesh: ShadowboxMesh
    bbox: BoundingBox | None


class TripoSRPipeline:
    """TripoSRによる3Dメッシュ生成パイプライン。

    ShadowboxPipelineと同様のインターフェースを持ちますが、
    深度推定+クラスタリングの代わりにTripoSRで直接3Dメッシュを生成します。

    Note:
        このクラスを使用するには、triposr依存関係が必要です:
        pip install shadowbox[triposr]

    Example:
        >>> from shadowbox.triposr import TripoSRPipeline, TripoSRSettings
        >>> settings = TripoSRSettings()
        >>> pipeline = TripoSRPipeline(settings)
        >>> result = pipeline.process(image)
    """

    def __init__(
        self,
        settings: TripoSRSettings,
        render_settings: RenderSettings | None = None,
    ) -> None:
        """パイプラインを初期化。

        Args:
            settings: TripoSR設定。
            render_settings: レンダリング設定（フレーム生成に使用）。
        """
        self._settings = settings
        self._render_settings = render_settings or RenderSettings()
        self._generator = TripoSRGenerator(settings)

    def process(
        self,
        image: str | Path | Image.Image | NDArray,
        bbox: BoundingBox | None = None,
        include_frame: bool = True,
    ) -> TripoSRPipelineResult:
        """画像を処理して3Dメッシュを生成。

        Args:
            image: 入力画像（パス、PIL Image、またはNumPy配列）。
            bbox: イラスト領域のバウンディングボックス（Noneの場合は画像全体）。
            include_frame: フレームを含めるかどうか。

        Returns:
            TripoSRPipelineResult: 生成結果。
        """
        # 画像をロード
        if isinstance(image, (str, Path)):
            pil_image = load_image(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        original_array = image_to_array(pil_image)

        # バウンディングボックスでクロップ
        if bbox is not None:
            cropped_image = crop_image(pil_image, bbox.x, bbox.y, bbox.width, bbox.height)
        else:
            cropped_image = pil_image

        # TripoSRで3Dメッシュを生成
        print("TripoSRで3Dメッシュを生成中...")
        mesh = self._generator.generate(cropped_image)
        print("3Dメッシュ生成完了")

        # フレームを追加
        if include_frame:
            mesh = self._add_frame_to_mesh(mesh)

        return TripoSRPipelineResult(
            original_image=original_array,
            mesh=mesh,
            bbox=bbox,
        )

    def _add_frame_to_mesh(self, mesh: ShadowboxMesh) -> ShadowboxMesh:
        """メッシュにフレームを追加。

        Args:
            mesh: TripoSRで生成されたメッシュ。

        Returns:
            フレームを追加したShadowboxMesh。
        """
        # メッシュのバウンズからZ範囲を取得
        z_min = mesh.bounds[4]  # min_z
        z_max = mesh.bounds[5]  # max_z

        # RenderSettingsのframe_wall_modeを活用
        if self._render_settings.frame_wall_mode == "none":
            config = FrameConfig(z_front=z_max)
        else:
            config = FrameConfig(z_front=z_max, z_back=z_min)

        frame = create_frame(config)
        bounds = calculate_bounds(mesh.layers, frame)

        return ShadowboxMesh(
            layers=mesh.layers,
            frame=frame,
            bounds=bounds,
        )

    @property
    def settings(self) -> TripoSRSettings:
        """現在の設定を取得。"""
        return self._settings

    @property
    def render_settings(self) -> RenderSettings:
        """レンダリング設定を取得。"""
        return self._render_settings
