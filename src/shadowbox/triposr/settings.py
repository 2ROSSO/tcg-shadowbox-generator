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

    Example:
        >>> # デフォルト設定
        >>> settings = TripoSRSettings()
        >>>
        >>> # 高解像度メッシュ生成
        >>> settings = TripoSRSettings(mc_resolution=512)
        >>>
        >>> # メモリ節約モード
        >>> settings = TripoSRSettings(chunk_size=4096, mc_resolution=128)
    """

    model_id: str = "stabilityai/TripoSR"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    chunk_size: int = 8192
    mc_resolution: int = 256
    remove_background: bool = True
    foreground_ratio: float = 0.85
