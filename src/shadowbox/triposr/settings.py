"""TripoSR設定モジュール。

TripoSR 3Dメッシュ生成器の設定を管理します。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TripoSRSettings:
    """TripoSR 3Dメッシュ生成器の設定。

    Stability AIのTripoSRモデルを使用して、
    単一画像から3Dメッシュを生成する際の設定を管理します。

    Attributes:
        model_id: Hugging Faceのモデル識別子。
        device: 推論を実行するデバイス。
            - "auto": 自動検出 (CUDA > CPU)
            - "cpu": CPU使用
            - "cuda": NVIDIA GPU使用
        chunk_size: メモリ効率化のためのチャンクサイズ。
            小さい値はメモリ使用量を削減するが遅くなる。
        mc_resolution: Marching Cubes法の解像度。
            高い値はより詳細なメッシュを生成するが、
            処理時間とメモリ使用量が増加する。
        remove_background: 入力画像から背景を除去するかどうか。
        foreground_ratio: 前景領域の比率（0.0-1.0）。
            背景除去後、この比率で前景がクロップされる。
        depth_resolution: 深度マップ復元時の解像度 (height, width)。
            split_by_depth使用時に適用。
        depth_fill_holes: 深度マップの穴埋め処理を行うかどうか。
        depth_fill_method: 穴埋めの方法。
            - "interpolate": OpenCV inpaintingによる補間
            - "max_depth": 最大深度値で埋める
        face_assignment_method: メッシュ分割時の面のレイヤー割り当て方法。
            - "centroid": 面の重心のZ座標で決定
            - "majority": 頂点の多数決で決定

    Example:
        >>> # デフォルト設定
        >>> settings = TripoSRSettings()
        >>>
        >>> # 高解像度メッシュ生成
        >>> settings = TripoSRSettings(mc_resolution=512)
        >>>
        >>> # メモリ節約モード
        >>> settings = TripoSRSettings(chunk_size=4096, mc_resolution=128)
        >>>
        >>> # 深度ベース分割を高解像度で実行
        >>> settings = TripoSRSettings(depth_resolution=(1024, 1024))
    """

    model_id: str = "stabilityai/TripoSR"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    chunk_size: int = 8192
    mc_resolution: int = 256
    remove_background: bool = True
    foreground_ratio: float = 0.85

    # 深度復元設定（split_by_depth使用時）
    depth_resolution: tuple[int, int] = (512, 512)
    depth_fill_holes: bool = True
    depth_fill_method: Literal["interpolate", "max_depth"] = "interpolate"

    # メッシュ分割設定
    face_assignment_method: Literal["centroid", "majority"] = "centroid"
