"""3Dレンダリングモジュール。

このモジュールは、Vedoを使用してシャドーボックスメッシュを
インタラクティブに3D表示する機能を提供します。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from shadowbox.core.mesh import ShadowboxMesh


@dataclass
class RenderOptions:
    """レンダリングオプション。

    Attributes:
        background_color: 背景色 (R, G, B) 各0-255。
        point_size: ポイントのサイズ。
        show_axes: 軸を表示するかどうか。
        show_frame: フレームを表示するかどうか。
        window_size: ウィンドウサイズ (幅, 高さ)。
        title: ウィンドウタイトル。
        interactive: インタラクティブモードを有効にするか。
        layer_opacity: レイヤーの不透明度 (0.0-1.0)。
    """

    background_color: Tuple[int, int, int] = (30, 30, 30)
    point_size: float = 3.0
    show_axes: bool = False
    show_frame: bool = True
    window_size: Tuple[int, int] = (1200, 800)
    title: str = "Shadowbox 3D Viewer"
    interactive: bool = True
    layer_opacity: float = 1.0


class ShadowboxRenderer:
    """シャドーボックスの3Dレンダラー。

    Vedoを使用してShadowboxMeshをインタラクティブに
    3D表示します。マウスでの回転・ズーム・パンに対応。

    Attributes:
        options: レンダリングオプション。

    Example:
        >>> from shadowbox.core.mesh import ShadowboxMesh
        >>> renderer = ShadowboxRenderer()
        >>> renderer.render(mesh)
    """

    def __init__(self, options: Optional[RenderOptions] = None) -> None:
        """レンダラーを初期化。

        Args:
            options: レンダリングオプション。Noneの場合はデフォルト。
        """
        self._options = options or RenderOptions()
        self._plotter = None

    def render(
        self,
        mesh: ShadowboxMesh,
        show: bool = True,
    ) -> "vedo.Plotter":
        """シャドーボックスメッシュを3Dレンダリング。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            show: 即座に表示するかどうか。

        Returns:
            VedoのPlotterオブジェクト。

        Example:
            >>> renderer = ShadowboxRenderer()
            >>> plotter = renderer.render(mesh, show=True)
        """
        import vedo

        # プロッターを作成
        self._plotter = vedo.Plotter(
            size=self._options.window_size,
            title=self._options.title,
            bg=self._normalize_color(self._options.background_color),
        )

        actors = []

        # 各レイヤーをポイントクラウドとして追加
        for layer in mesh.layers:
            if len(layer.vertices) == 0:
                continue

            # ポイントクラウドを作成
            points = vedo.Points(layer.vertices, r=self._options.point_size)

            # 色を設定（Vedoはpointdata["RGB"]で頂点ごとの色を設定）
            points.pointdata["RGB"] = layer.colors
            points.pointdata.select("RGB")
            points.alpha(self._options.layer_opacity)
            actors.append(points)

        # フレームを追加
        if self._options.show_frame and mesh.frame is not None:
            frame_mesh = self._create_frame_mesh(mesh.frame)
            if frame_mesh is not None:
                actors.append(frame_mesh)

        # 軸を表示
        if self._options.show_axes:
            axes = vedo.Axes(
                actors[0] if actors else None,
                xtitle="X",
                ytitle="Y",
                ztitle="Z (深度)",
            )
            actors.append(axes)

        # アクターを追加
        self._plotter.add(actors)

        # カメラ位置を調整（正面から少し斜めに）
        self._plotter.camera.SetPosition(0, 0, 3)
        self._plotter.camera.SetFocalPoint(0, 0, -0.5)
        self._plotter.camera.SetViewUp(0, 1, 0)

        if show:
            self._plotter.show(interactive=self._options.interactive)

        return self._plotter

    def render_layers_separately(
        self,
        mesh: ShadowboxMesh,
        gap_multiplier: float = 2.0,
    ) -> "vedo.Plotter":
        """レイヤーを分離して表示。

        デバッグ用に、各レイヤーを本来の位置より離して
        表示することで、レイヤー構造を確認しやすくします。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            gap_multiplier: レイヤー間隔の倍率。

        Returns:
            VedoのPlotterオブジェクト。

        Example:
            >>> renderer = ShadowboxRenderer()
            >>> plotter = renderer.render_layers_separately(mesh, gap_multiplier=3.0)
        """
        import vedo

        self._plotter = vedo.Plotter(
            size=self._options.window_size,
            title=f"{self._options.title} (レイヤー分離表示)",
            bg=self._normalize_color(self._options.background_color),
        )

        actors = []

        for i, layer in enumerate(mesh.layers):
            if len(layer.vertices) == 0:
                continue

            # Z座標を調整（より離す）
            vertices = layer.vertices.copy()
            original_z = layer.z_position
            vertices[:, 2] = original_z * gap_multiplier

            points = vedo.Points(vertices, r=self._options.point_size)
            points.pointdata["RGB"] = layer.colors
            points.pointdata.select("RGB")
            points.alpha(self._options.layer_opacity)
            actors.append(points)

            # レイヤー番号のラベルを追加
            label_pos = [0, 1.2, original_z * gap_multiplier]
            label = vedo.Text3D(
                f"Layer {i}",
                pos=label_pos,
                s=0.08,
                c="white",
                justify="center",
            )
            actors.append(label)

        self._plotter.add(actors)
        self._plotter.show(interactive=self._options.interactive)

        return self._plotter

    def export_screenshot(
        self,
        mesh: ShadowboxMesh,
        filename: str,
        size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """レンダリング結果をスクリーンショットとして保存。

        Args:
            mesh: レンダリングするシャドーボックスメッシュ。
            filename: 保存先のファイルパス。
            size: 画像サイズ。Noneの場合はウィンドウサイズ。

        Example:
            >>> renderer = ShadowboxRenderer()
            >>> renderer.export_screenshot(mesh, "shadowbox.png")
        """
        import vedo

        # オフスクリーンレンダリング
        if size:
            orig_size = self._options.window_size
            self._options.window_size = size

        plotter = vedo.Plotter(
            size=self._options.window_size,
            offscreen=True,
            bg=self._normalize_color(self._options.background_color),
        )

        actors = []
        for layer in mesh.layers:
            if len(layer.vertices) == 0:
                continue

            points = vedo.Points(layer.vertices, r=self._options.point_size)
            points.pointdata["RGB"] = layer.colors
            points.pointdata.select("RGB")
            actors.append(points)

        if self._options.show_frame and mesh.frame is not None:
            frame_mesh = self._create_frame_mesh(mesh.frame)
            if frame_mesh is not None:
                actors.append(frame_mesh)

        plotter.add(actors)
        plotter.camera.SetPosition(0, 0, 3)
        plotter.camera.SetFocalPoint(0, 0, -0.5)
        plotter.camera.SetViewUp(0, 1, 0)

        plotter.screenshot(filename)
        plotter.close()

        if size:
            self._options.window_size = orig_size

    def _create_frame_mesh(self, frame: "FrameMesh") -> Optional["vedo.Mesh"]:
        """フレームメッシュをVedoメッシュに変換。

        Args:
            frame: フレームメッシュデータ。

        Returns:
            Vedoメッシュオブジェクト。頂点がない場合はNone。
        """
        import vedo

        if len(frame.vertices) == 0:
            return None

        color_normalized = frame.color.astype(np.float32) / 255.0

        mesh = vedo.Mesh([frame.vertices, frame.faces])
        mesh.c(color_normalized)
        mesh.alpha(0.9)

        return mesh

    def _normalize_color(
        self,
        color: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """色を0-255から0-1の範囲に正規化。

        Args:
            color: RGB色 (0-255)。

        Returns:
            正規化されたRGB色 (0-1)。
        """
        return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def close(self) -> None:
        """プロッターを閉じる。"""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None


def render_shadowbox(
    mesh: ShadowboxMesh,
    options: Optional[RenderOptions] = None,
) -> "vedo.Plotter":
    """シャドーボックスを簡単にレンダリングするユーティリティ関数。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        options: レンダリングオプション。Noneの場合はデフォルト。

    Returns:
        VedoのPlotterオブジェクト。

    Example:
        >>> from shadowbox import create_pipeline
        >>> from shadowbox.visualization import render_shadowbox
        >>>
        >>> pipeline = create_pipeline(use_mock_depth=True)
        >>> result = pipeline.process(image)
        >>> render_shadowbox(result.mesh)
    """
    renderer = ShadowboxRenderer(options)
    return renderer.render(mesh)


def render_layers_exploded(
    mesh: ShadowboxMesh,
    gap_multiplier: float = 2.5,
    options: Optional[RenderOptions] = None,
) -> "vedo.Plotter":
    """レイヤーを分解して表示するユーティリティ関数。

    Args:
        mesh: レンダリングするシャドーボックスメッシュ。
        gap_multiplier: レイヤー間隔の倍率。
        options: レンダリングオプション。Noneの場合はデフォルト。

    Returns:
        VedoのPlotterオブジェクト。

    Example:
        >>> render_layers_exploded(mesh, gap_multiplier=3.0)
    """
    renderer = ShadowboxRenderer(options)
    return renderer.render_layers_separately(mesh, gap_multiplier)
