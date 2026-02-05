"""共通パイプライン基底クラスモジュール。

このモジュールは、すべてのパイプライン結果で共通する
基底クラスを提供します。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shadowbox.config.template import BoundingBox
from shadowbox.core.mesh import ShadowboxMesh


@dataclass
class BasePipelineResult:
    """パイプライン結果の基底クラス。

    すべてのパイプライン結果で共通するフィールドを定義します。

    Attributes:
        original_image: 元の入力画像（NumPy配列）。
        mesh: 生成された3Dメッシュ。
        bbox: 使用されたバウンディングボックス（クロップした場合）。
    """

    original_image: NDArray[np.uint8]
    mesh: ShadowboxMesh
    bbox: BoundingBox | None
