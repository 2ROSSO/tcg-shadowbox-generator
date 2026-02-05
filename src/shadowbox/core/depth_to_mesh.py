"""深度マップからメッシュへの共通変換モジュール。

深度推定モードとTripoSRモードの両方で使用される、
depth_map + image → ShadowboxMesh の共通処理を提供します。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shadowbox.config.template import BoundingBox
from shadowbox.core.clustering import LayerClustererProtocol
from shadowbox.core.mesh import MeshGeneratorProtocol, ShadowboxMesh


@dataclass(frozen=True)
class DepthToMeshInput:
    """深度復元後の統一中間データ。

    Attributes:
        cropped_image: クロップされた画像 (H, W, 3)。
        depth_map: 正規化された深度マップ (H, W), [0,1]。
        original_image: 元の入力画像（カードフレーム統合時に使用）。
        bbox: バウンディングボックス（カードフレーム統合時に使用）。
    """

    cropped_image: NDArray[np.uint8]
    depth_map: NDArray[np.float32]
    original_image: NDArray[np.uint8] | None = None
    bbox: BoundingBox | None = None


@dataclass
class DepthToMeshResult:
    """深度→メッシュ変換の結果。

    Attributes:
        mesh: 生成されたシャドーボックスメッシュ。
        labels: 各ピクセルのレイヤーインデックス。
        centroids: 各レイヤーの深度セントロイド。
        optimal_k: 使用されたレイヤー数。
    """

    mesh: ShadowboxMesh
    labels: NDArray[np.int32]
    centroids: NDArray[np.float32]
    optimal_k: int


class DepthToMeshProcessor:
    """depth_map + image → ShadowboxMesh の共通処理器。

    DepthPipelineとTripoSRPipelineの両方から使用され、
    クラスタリング→メッシュ生成の共通処理を提供します。

    Attributes:
        clusterer: レイヤークラスタラー。
        mesh_generator: メッシュジェネレーター。

    Example:
        >>> processor = DepthToMeshProcessor(clusterer, mesh_generator)
        >>> input_data = DepthToMeshInput(cropped_image=img, depth_map=depth)
        >>> result = processor.process(input_data, k=5)
    """

    def __init__(
        self,
        clusterer: LayerClustererProtocol,
        mesh_generator: MeshGeneratorProtocol,
    ) -> None:
        """プロセッサを初期化。

        Args:
            clusterer: レイヤークラスタラーインスタンス。
            mesh_generator: メッシュジェネレーターインスタンス。
        """
        self._clusterer = clusterer
        self._mesh_generator = mesh_generator

    def process(
        self,
        input_data: DepthToMeshInput,
        k: int | None = None,
        include_frame: bool = True,
        include_card_frame: bool = False,
        use_raw_depth: bool = False,
        depth_scale: float = 1.0,
    ) -> DepthToMeshResult:
        """深度マップからシャドーボックスメッシュを生成。

        Args:
            input_data: 入力データ（画像と深度マップ）。
            k: レイヤー数（Noneの場合は自動探索）。use_raw_depth=Trueの場合は無視。
            include_frame: フレームを含めるかどうか。
            include_card_frame: カードのフレーム部分を含めるかどうか。
                Trueの場合、イラスト領域外のピクセルを最前面の深度で統合。
            use_raw_depth: 生の深度値を使用するかどうか。
                Trueの場合、クラスタリングをスキップし、各ピクセルの深度を
                直接Z座標として使用。
            depth_scale: 生深度モード時の深度スケール。

        Returns:
            DepthToMeshResult: メッシュ生成結果。
        """
        # カードフレーム統合時の画像選択
        has_card_frame_data = (
            include_card_frame
            and input_data.original_image is not None
            and input_data.bbox is not None
        )
        if has_card_frame_data:
            cropped_array = input_data.original_image
        else:
            cropped_array = input_data.cropped_image

        depth_map = input_data.depth_map

        if use_raw_depth:
            # 生深度モード: クラスタリングをスキップ
            labels = np.zeros_like(depth_map, dtype=np.int32)
            centroids = np.array([0.5], dtype=np.float32)
            optimal_k = 1

            mesh = self._mesh_generator.generate_raw_depth(
                cropped_array,
                depth_map,
                include_frame=include_frame,
                depth_scale=depth_scale,
            )
        else:
            # クラスタリングモード
            if k is None:
                optimal_k = self._clusterer.find_optimal_k(depth_map)
            else:
                optimal_k = k

            labels, centroids = self._clusterer.cluster(depth_map, optimal_k)

            # カードフレーム統合時: フレーム領域を特別ラベルでマーク
            if include_card_frame and input_data.bbox is not None:
                bbox = input_data.bbox
                frame_mask = np.ones_like(labels, dtype=bool)
                frame_mask[
                    bbox.y : bbox.y + bbox.height,
                    bbox.x : bbox.x + bbox.width,
                ] = False
                labels[frame_mask] = -1

            mesh = self._mesh_generator.generate(
                cropped_array,
                labels,
                centroids,
                include_frame=include_frame,
            )

        return DepthToMeshResult(
            mesh=mesh,
            labels=labels,
            centroids=centroids,
            optimal_k=optimal_k,
        )
