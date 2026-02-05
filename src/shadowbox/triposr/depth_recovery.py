"""深度マップ復元モジュール。

TripoSRで生成した3Dメッシュを画像平面に投影して
深度マップを復元する機能を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DepthRecoverySettings:
    """深度マップ復元の設定。

    Attributes:
        resolution: 出力深度マップの解像度 (height, width)。
        fill_holes: 穴埋め処理を行うかどうか。
        hole_fill_method: 穴埋めの方法。
            - "interpolate": OpenCV inpaintingによる補間
            - "max_depth": 最大深度値で埋める
    """

    resolution: tuple[int, int] = (512, 512)
    fill_holes: bool = True
    hole_fill_method: Literal["interpolate", "max_depth"] = "interpolate"


class MeshDepthExtractorProtocol(Protocol):
    """メッシュ深度抽出器のプロトコル（DIインターフェース）。

    このプロトコルを実装することで、異なるレンダリングバックエンドを
    使用できます（pyrender, trimesh ray casting など）。
    """

    def extract_depth(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        resolution: tuple[int, int],
    ) -> NDArray[np.float32]:
        """メッシュを投影して深度マップを生成。

        Args:
            vertices: メッシュの頂点座標。shape (N, 3)。
            faces: 三角形面のインデックス。shape (M, 3)。
            resolution: 出力解像度 (height, width)。

        Returns:
            depth_map: 正規化された深度マップ [0, 1]。shape (H, W)。
                見えないピクセルはNaN。
        """
        ...


class PyRenderDepthExtractor:
    """pyrenderを使用したオフスクリーン深度抽出器。

    OpenGLベースのオフスクリーンレンダリングで深度バッファを取得します。

    Note:
        pyrenderが利用不可の場合は、TrimeshRayCastingExtractorに
        フォールバックすることを推奨します。
    """

    def __init__(self) -> None:
        """深度抽出器を初期化。"""
        self._renderer = None

    def extract_depth(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        resolution: tuple[int, int],
    ) -> NDArray[np.float32]:
        """メッシュを投影して深度マップを生成。

        Args:
            vertices: メッシュの頂点座標。shape (N, 3)。
            faces: 三角形面のインデックス。shape (M, 3)。
            resolution: 出力解像度 (height, width)。

        Returns:
            depth_map: 正規化された深度マップ [0, 1]。shape (H, W)。
                見えないピクセルはNaN。
        """
        try:
            import pyrender
            import trimesh
        except ImportError as e:
            raise ImportError(
                "深度抽出にはpyrenderが必要です。\n"
                "インストール: pip install pyrender"
            ) from e

        h, w = resolution

        # trimeshメッシュを作成
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # pyrenderメッシュに変換
        pr_mesh = pyrender.Mesh.from_trimesh(mesh)

        # シーンを作成
        scene = pyrender.Scene()
        scene.add(pr_mesh)

        # カメラを設定（正射影、正面から）
        # メッシュは[-1, 1]に正規化されているため、それに合わせる
        camera = pyrender.OrthographicCamera(
            xmag=1.0,
            ymag=1.0,
            znear=0.01,
            zfar=10.0,
        )

        # カメラをZ軸正方向に配置（正面から見る）
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],  # Z=3の位置から見る
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)

        # オフスクリーンレンダリング
        renderer = pyrender.OffscreenRenderer(w, h)
        try:
            _, depth = renderer.render(scene)
        finally:
            renderer.delete()

        # 深度値を処理
        # pyrender は深度バッファを返す（0=近い、大きい値=遠い）
        # 背景（レンダリングされなかった部分）は0になる

        # 背景をNaNに変換
        depth_map = depth.astype(np.float32)
        depth_map[depth == 0] = np.nan

        # 正規化 [0, 1]（0が手前、1が奥）
        valid_mask = ~np.isnan(depth_map)
        if np.any(valid_mask):
            min_depth = np.nanmin(depth_map)
            max_depth = np.nanmax(depth_map)
            if max_depth > min_depth:
                depth_map = (depth_map - min_depth) / (max_depth - min_depth)
            else:
                depth_map[valid_mask] = 0.5

        return depth_map


class TrimeshRayCastingExtractor:
    """trimeshのレイキャスティングを使用した深度抽出器。

    OpenGLが利用不可の環境でのフォールバック用。
    pyrenderより遅いが、CPU環境で動作します。
    """

    def extract_depth(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        resolution: tuple[int, int],
    ) -> NDArray[np.float32]:
        """メッシュを投影して深度マップを生成（レイキャスティング）。

        Args:
            vertices: メッシュの頂点座標。shape (N, 3)。
            faces: 三角形面のインデックス。shape (M, 3)。
            resolution: 出力解像度 (height, width)。

        Returns:
            depth_map: 正規化された深度マップ [0, 1]。shape (H, W)。
                見えないピクセルはNaN。
        """
        try:
            import trimesh
        except ImportError as e:
            raise ImportError(
                "深度抽出にはtrimeshが必要です。\n"
                "インストール: pip install trimesh"
            ) from e

        h, w = resolution

        # trimeshメッシュを作成
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # レイの原点と方向を作成
        # 正射影: 各ピクセルから-Z方向にレイを飛ばす
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # ピクセル座標を[-1, 1]に正規化
        ray_x = (x_coords.flatten() / (w - 1)) * 2 - 1 if w > 1 else np.zeros(h * w)
        ray_y = -((y_coords.flatten() / (h - 1)) * 2 - 1) if h > 1 else np.zeros(h * w)

        # レイの原点（Z=5から）と方向（-Z方向）
        ray_origins = np.column_stack([
            ray_x,
            ray_y,
            np.full(h * w, 5.0),
        ]).astype(np.float32)

        ray_directions = np.tile(
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            (h * w, 1),
        )

        # レイキャスティング
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )

        # 深度マップを初期化（NaN）
        depth_map = np.full((h * w,), np.nan, dtype=np.float32)

        if len(locations) > 0:
            # 各レイで最も近い交点のZ座標を取得
            # 同じレイに複数の交点がある場合は最も近いもの（Z座標が大きい）を選択
            for i in range(h * w):
                ray_hits = locations[index_ray == i]
                if len(ray_hits) > 0:
                    # Z座標が最も大きい（カメラに近い）点を選択
                    depth_map[i] = ray_hits[:, 2].max()

        depth_map = depth_map.reshape((h, w))

        # 正規化 [0, 1]（0が手前、1が奥）
        valid_mask = ~np.isnan(depth_map)
        if np.any(valid_mask):
            min_depth = np.nanmin(depth_map)
            max_depth = np.nanmax(depth_map)
            if max_depth > min_depth:
                # Z座標が大きい=手前なので、反転して正規化
                depth_map = 1.0 - (depth_map - min_depth) / (max_depth - min_depth)
            else:
                depth_map[valid_mask] = 0.5

        return depth_map


def fill_depth_holes(
    depth_map: NDArray[np.float32],
    method: Literal["interpolate", "max_depth"] = "interpolate",
) -> NDArray[np.float32]:
    """深度マップの穴（NaN）を埋める。

    Args:
        depth_map: 穴（NaN）を含む深度マップ。shape (H, W)。
        method: 穴埋めの方法。
            - "interpolate": OpenCV inpaintingによる補間
            - "max_depth": 最大深度値（最も奥）で埋める

    Returns:
        filled_depth_map: 穴埋め後の深度マップ。shape (H, W)。
    """
    filled = depth_map.copy()
    mask = np.isnan(filled)

    if not np.any(mask):
        return filled

    if method == "max_depth":
        # NaNを最大深度値（最も奥）で埋める
        max_depth = np.nanmax(filled) if np.any(~mask) else 1.0
        filled[mask] = max_depth
    elif method == "interpolate":
        try:
            import cv2
        except ImportError:
            # OpenCVがない場合はmax_depthにフォールバック
            max_depth = np.nanmax(filled) if np.any(~mask) else 1.0
            filled[mask] = max_depth
            return filled

        # OpenCV inpaintingを使用
        # 深度値を0-255に変換
        valid_mask = ~mask
        if np.any(valid_mask):
            min_val = np.nanmin(filled)
            max_val = np.nanmax(filled)
            if max_val > min_val:
                depth_uint8 = ((filled - min_val) / (max_val - min_val) * 255).astype(
                    np.uint8
                )
            else:
                depth_uint8 = np.full_like(filled, 128, dtype=np.uint8)
        else:
            depth_uint8 = np.full_like(filled, 128, dtype=np.uint8)

        # マスクを作成（穴=255）
        inpaint_mask = mask.astype(np.uint8) * 255

        # inpainting
        inpainted = cv2.inpaint(depth_uint8, inpaint_mask, 3, cv2.INPAINT_TELEA)

        # 元のスケールに戻す
        if np.any(valid_mask):
            min_val = np.nanmin(depth_map)
            max_val = np.nanmax(depth_map)
            if max_val > min_val:
                filled = inpainted.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
            else:
                filled = np.full_like(depth_map, 0.5, dtype=np.float32)
        else:
            filled = inpainted.astype(np.float32) / 255.0

    return filled


def create_depth_extractor(use_pyrender: bool = True) -> MeshDepthExtractorProtocol:
    """深度抽出器を作成。

    Args:
        use_pyrender: pyrenderを使用するかどうか。
            Falseの場合、trimeshのレイキャスティングを使用。

    Returns:
        MeshDepthExtractorProtocol: 深度抽出器。
    """
    if use_pyrender:
        try:
            import pyrender  # noqa: F401
            return PyRenderDepthExtractor()
        except ImportError:
            pass

    return TrimeshRayCastingExtractor()
