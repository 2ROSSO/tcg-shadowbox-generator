"""メッシュ分割モジュール。

深度クラスタリングに基づいてTripoSRメッシュをレイヤーに分割する
機能を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from shadowbox.core.clustering import LayerClustererProtocol
from shadowbox.core.mesh import LayerMesh, ShadowboxMesh
from shadowbox.triposr.depth_recovery import (
    DepthRecoverySettings,
    MeshDepthExtractorProtocol,
    create_depth_extractor,
    fill_depth_holes,
)

if TYPE_CHECKING:
    pass


@dataclass
class MeshSplitResult:
    """メッシュ分割の結果。

    Attributes:
        layer_meshes: レイヤーごとに分割されたメッシュのリスト。
        depth_map: 復元された深度マップ。
        labels: 深度マップのクラスタラベル。
        centroids: クラスタのセントロイド（深度値）。
    """

    layer_meshes: list[LayerMesh]
    depth_map: NDArray[np.float32]
    labels: NDArray[np.int32]
    centroids: NDArray[np.float32]


class DepthBasedMeshSplitter:
    """深度クラスタリングに基づくメッシュ分割器。

    TripoSRで生成した3Dメッシュを画像平面に投影して深度マップを復元し、
    既存のクラスタリング処理を適用してレイヤーに分割します。

    Example:
        >>> from shadowbox.core.clustering import KMeansLayerClusterer
        >>> from shadowbox.config.settings import ClusteringSettings
        >>> clusterer = KMeansLayerClusterer(ClusteringSettings())
        >>> splitter = DepthBasedMeshSplitter(clusterer=clusterer)
        >>> result = splitter.split(vertices, faces, colors, k=5)
    """

    def __init__(
        self,
        depth_extractor: MeshDepthExtractorProtocol | None = None,
        clusterer: LayerClustererProtocol | None = None,
        settings: DepthRecoverySettings | None = None,
    ) -> None:
        """メッシュ分割器を初期化。

        Args:
            depth_extractor: 深度抽出器。Noneの場合は自動選択。
            clusterer: クラスタラー。Noneの場合はデフォルトを使用。
            settings: 深度復元設定。Noneの場合はデフォルト。
        """
        self._depth_extractor = depth_extractor or create_depth_extractor()
        self._clusterer = clusterer
        self._settings = settings or DepthRecoverySettings()

    def split(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        colors: NDArray[np.uint8],
        k: int | None = None,
        face_assignment_method: Literal["centroid", "majority"] = "centroid",
    ) -> MeshSplitResult:
        """メッシュを深度ベースでレイヤーに分割。

        Args:
            vertices: メッシュの頂点座標。shape (N, 3)。
            faces: 三角形面のインデックス。shape (M, 3)。
            colors: 各頂点の色。shape (N, 3)。
            k: レイヤー数。Noneの場合は自動決定。
            face_assignment_method: 面のレイヤー割り当て方法。
                - "centroid": 面の重心のZ座標で決定
                - "majority": 頂点の多数決で決定

        Returns:
            MeshSplitResult: 分割結果。
        """
        # 1. 深度マップを復元
        depth_map = self._depth_extractor.extract_depth(
            vertices, faces, self._settings.resolution
        )

        # 2. 穴埋め処理
        if self._settings.fill_holes:
            depth_map = fill_depth_holes(depth_map, self._settings.hole_fill_method)

        # 3. クラスタリング
        if self._clusterer is None:
            from shadowbox.config.settings import ClusteringSettings
            from shadowbox.core.clustering import KMeansLayerClusterer
            self._clusterer = KMeansLayerClusterer(ClusteringSettings())

        if k is None:
            k = self._clusterer.find_optimal_k(depth_map)

        labels, centroids = self._clusterer.cluster(depth_map, k)

        # 4. 頂点にレイヤーラベルを付与
        vertex_labels = self._assign_vertex_labels(
            vertices, depth_map, labels, self._settings.resolution
        )

        # 5. 面をレイヤーに分割
        face_labels = self._assign_face_labels(
            faces, vertices, vertex_labels, centroids, face_assignment_method
        )

        # 6. レイヤーごとにメッシュを作成
        layer_meshes = self._create_layer_meshes(
            vertices, faces, colors, face_labels, centroids
        )

        return MeshSplitResult(
            layer_meshes=layer_meshes,
            depth_map=depth_map,
            labels=labels,
            centroids=centroids,
        )

    def _assign_vertex_labels(
        self,
        vertices: NDArray[np.float32],
        depth_map: NDArray[np.float32],
        labels: NDArray[np.int32],
        resolution: tuple[int, int],
    ) -> NDArray[np.int32]:
        """頂点にクラスタラベルを割り当て。

        Args:
            vertices: メッシュの頂点座標。shape (N, 3)。
            depth_map: 深度マップ（参照用）。shape (H, W)。
            labels: 深度マップのクラスタラベル。shape (H, W)。
            resolution: 深度マップの解像度 (H, W)。

        Returns:
            vertex_labels: 各頂点のレイヤーラベル。shape (N,)。
        """
        h, w = resolution
        n_vertices = len(vertices)

        # 頂点のXY座標をピクセル座標に変換
        pixel_coords = compute_vertex_pixel_mapping(vertices, resolution)

        # ピクセル座標が範囲内かチェック
        valid_y = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h)
        valid_x = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w)
        valid = valid_y & valid_x

        # ラベルを割り当て
        vertex_labels = np.zeros(n_vertices, dtype=np.int32)
        vertex_labels[valid] = labels[
            pixel_coords[valid, 0], pixel_coords[valid, 1]
        ]

        # 範囲外の頂点は最も近いセントロイドで割り当て
        # （Z座標ベースで決定）
        if not np.all(valid):
            invalid_mask = ~valid
            invalid_z = vertices[invalid_mask, 2]

            # 深度マップの統計から推定
            unique_labels = np.unique(labels)
            centroids_z = []
            for label in unique_labels:
                # ラベルに対応する頂点のZ座標の平均を使用
                if np.any(valid):
                    label_vertices = vertices[valid][vertex_labels[valid] == label]
                    if len(label_vertices) > 0:
                        centroids_z.append(label_vertices[:, 2].mean())
                    else:
                        centroids_z.append(0.0)
                else:
                    centroids_z.append(0.0)

            centroids_z = np.array(centroids_z)

            # 各無効頂点を最も近いセントロイドに割り当て
            distances = np.abs(invalid_z[:, None] - centroids_z[None, :])
            vertex_labels[invalid_mask] = unique_labels[np.argmin(distances, axis=1)]

        return vertex_labels

    def _assign_face_labels(
        self,
        faces: NDArray[np.int32],
        vertices: NDArray[np.float32],
        vertex_labels: NDArray[np.int32],
        centroids: NDArray[np.float32],
        method: Literal["centroid", "majority"],
    ) -> NDArray[np.int32]:
        """面にレイヤーラベルを割り当て。

        Args:
            faces: 三角形面のインデックス。shape (M, 3)。
            vertices: メッシュの頂点座標。shape (N, 3)。
            vertex_labels: 各頂点のレイヤーラベル。shape (N,)。
            centroids: クラスタのセントロイド（深度値）。shape (k,)。
            method: 割り当て方法。

        Returns:
            face_labels: 各面のレイヤーラベル。shape (M,)。
        """
        if method == "centroid":
            return assign_face_labels_centroid(faces, vertices, centroids)
        else:
            return assign_face_labels_majority(faces, vertex_labels)

    def _create_layer_meshes(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        colors: NDArray[np.uint8],
        face_labels: NDArray[np.int32],
        centroids: NDArray[np.float32],
    ) -> list[LayerMesh]:
        """レイヤーごとにメッシュを作成。

        Args:
            vertices: メッシュの頂点座標。shape (N, 3)。
            faces: 三角形面のインデックス。shape (M, 3)。
            colors: 各頂点の色。shape (N, 3)。
            face_labels: 各面のレイヤーラベル。shape (M,)。
            centroids: クラスタのセントロイド（深度値）。shape (k,)。

        Returns:
            layer_meshes: レイヤーごとのLayerMeshリスト。
        """
        layer_meshes = []
        n_layers = len(centroids)

        for layer_idx in range(n_layers):
            # このレイヤーに属する面を取得
            layer_face_mask = face_labels == layer_idx
            layer_faces = faces[layer_face_mask]

            if len(layer_faces) == 0:
                # 空のレイヤー
                layer_meshes.append(
                    LayerMesh(
                        vertices=np.array([], dtype=np.float32).reshape(0, 3),
                        colors=np.array([], dtype=np.uint8).reshape(0, 3),
                        z_position=float(centroids[layer_idx]),
                        layer_index=layer_idx,
                        pixel_indices=np.array([], dtype=np.int32).reshape(0, 2),
                        faces=np.array([], dtype=np.int32).reshape(0, 3),
                    )
                )
                continue

            # このレイヤーで使用される頂点のインデックス
            used_vertex_indices = np.unique(layer_faces.flatten())

            # 頂点インデックスをリマップ
            vertex_remap = {old: new for new, old in enumerate(used_vertex_indices)}

            # 新しい頂点配列
            layer_vertices = vertices[used_vertex_indices]
            layer_colors = colors[used_vertex_indices]

            # 面のインデックスをリマップ
            remapped_faces = np.vectorize(vertex_remap.get)(layer_faces)

            # ピクセルインデックス（TripoSRメッシュでは未使用だが必須フィールド）
            pixel_indices = np.zeros((len(layer_vertices), 2), dtype=np.int32)

            layer_meshes.append(
                LayerMesh(
                    vertices=layer_vertices.astype(np.float32),
                    colors=layer_colors.astype(np.uint8),
                    z_position=float(centroids[layer_idx]),
                    layer_index=layer_idx,
                    pixel_indices=pixel_indices,
                    faces=remapped_faces.astype(np.int32),
                )
            )

        return layer_meshes


def compute_vertex_pixel_mapping(
    vertices: NDArray[np.float32],
    resolution: tuple[int, int],
) -> NDArray[np.int32]:
    """頂点座標をピクセル座標に変換。

    メッシュの頂点XY座標（[-1, 1]範囲）を
    画像のピクセル座標に変換します。

    Args:
        vertices: メッシュの頂点座標。shape (N, 3)。
        resolution: 出力解像度 (height, width)。

    Returns:
        pixel_coords: ピクセル座標 (row, col)。shape (N, 2)。
    """
    h, w = resolution

    # X: [-1, 1] → [0, W-1]
    pixel_x = ((vertices[:, 0] + 1) / 2 * (w - 1)).astype(np.int32)

    # Y: [-1, 1] → [H-1, 0] (Y軸反転：メッシュは上が+Y、画像は上が0)
    pixel_y = ((1 - vertices[:, 1]) / 2 * (h - 1)).astype(np.int32)

    return np.stack([pixel_y, pixel_x], axis=1)


def assign_face_labels_centroid(
    faces: NDArray[np.int32],
    vertices: NDArray[np.float32],
    centroids: NDArray[np.float32],
) -> NDArray[np.int32]:
    """面の重心Z座標でレイヤーを決定。

    Args:
        faces: 三角形面のインデックス。shape (M, 3)。
        vertices: メッシュの頂点座標。shape (N, 3)。
        centroids: クラスタのセントロイド（深度値）。shape (k,)。

    Returns:
        face_labels: 各面のレイヤーラベル。shape (M,)。
    """
    # 面の頂点座標を取得 (M, 3, 3)
    face_verts = vertices[faces]

    # 面の重心を計算 (M, 3)
    face_centroids = face_verts.mean(axis=1)

    # Z座標を取得
    face_depths = face_centroids[:, 2]

    # セントロイドとの距離で最も近いクラスタを選択
    # centroids は深度値（正規化された値）なので、
    # 頂点のZ座標を深度に変換する必要がある

    # Z座標の範囲を取得
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()

    if z_max > z_min:
        # Z座標を[0, 1]に正規化（大きいZ=手前=0、小さいZ=奥=1）
        normalized_depths = 1.0 - (face_depths - z_min) / (z_max - z_min)
    else:
        normalized_depths = np.full_like(face_depths, 0.5)

    # 各面を最も近いセントロイドに割り当て
    distances = np.abs(normalized_depths[:, None] - centroids[None, :])
    face_labels = np.argmin(distances, axis=1).astype(np.int32)

    return face_labels


def assign_face_labels_majority(
    faces: NDArray[np.int32],
    vertex_labels: NDArray[np.int32],
) -> NDArray[np.int32]:
    """面の頂点ラベルの多数決でレイヤーを決定。

    Args:
        faces: 三角形面のインデックス。shape (M, 3)。
        vertex_labels: 各頂点のレイヤーラベル。shape (N,)。

    Returns:
        face_labels: 各面のレイヤーラベル。shape (M,)。
    """
    # 各面の3頂点のラベルを取得 (M, 3)
    face_vertex_labels = vertex_labels[faces]

    # 多数決（モード）を計算
    # 3頂点なので、2つ以上が同じラベルならそれを選択
    # すべて異なる場合は最初の頂点のラベルを使用
    face_labels = np.zeros(len(faces), dtype=np.int32)

    for i, labels in enumerate(face_vertex_labels):
        unique, counts = np.unique(labels, return_counts=True)
        face_labels[i] = unique[np.argmax(counts)]

    return face_labels


def create_split_shadowbox_mesh(
    split_result: MeshSplitResult,
    original_bounds: tuple,
) -> ShadowboxMesh:
    """分割結果からShadowboxMeshを作成。

    Args:
        split_result: メッシュ分割結果。
        original_bounds: 元のメッシュのバウンディングボックス。

    Returns:
        ShadowboxMesh: 分割されたメッシュ。
    """
    return ShadowboxMesh(
        layers=split_result.layer_meshes,
        frame=None,
        bounds=original_bounds,
    )
