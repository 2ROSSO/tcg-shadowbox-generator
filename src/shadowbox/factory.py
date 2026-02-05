"""Pipeline factory module.

パイプラインを作成するためのファクトリ関数を提供します。
settings.model_modeに基づいて適切なパイプラインを返します。
"""

from shadowbox.config.settings import ShadowboxSettings


def create_pipeline(
    settings: ShadowboxSettings | None = None,
    use_mock_depth: bool = False,
):
    """設定に基づいてパイプラインを作成するファクトリ関数。

    依存性注入を内部で処理し、設定に応じた
    パイプラインインスタンスを返します。

    Args:
        settings: シャドーボックス設定。Noneの場合はデフォルト。
        use_mock_depth: テスト用にモック深度推定器を使用するかどうか。

    Returns:
        model_mode="depth"の場合: DepthPipelineインスタンス。
        model_mode="triposr"の場合: TripoSRPipelineインスタンス。

    Example:
        >>> # デフォルト設定（深度推定モード）
        >>> pipeline = create_pipeline()
        >>>
        >>> # TripoSRモード
        >>> settings = ShadowboxSettings(model_mode="triposr")
        >>> pipeline = create_pipeline(settings)
        >>>
        >>> # テスト用（モック深度推定）
        >>> pipeline = create_pipeline(use_mock_depth=True)
    """
    if settings is None:
        settings = ShadowboxSettings()

    # 共通コンポーネントを作成
    from shadowbox.core.clustering import KMeansLayerClusterer
    from shadowbox.core.depth_to_mesh import DepthToMeshProcessor
    from shadowbox.core.mesh import MeshGenerator

    clusterer = KMeansLayerClusterer(settings.clustering)
    mesh_generator = MeshGenerator(settings.render)
    depth_to_mesh = DepthToMeshProcessor(clusterer, mesh_generator)

    # TripoSRモードの場合
    if settings.model_mode == "triposr":
        from shadowbox.triposr import create_triposr_pipeline

        return create_triposr_pipeline(
            settings.triposr, settings.render, depth_to_mesh
        )

    # 深度推定モード（デフォルト）
    from shadowbox.config.loader import YAMLConfigLoader
    from shadowbox.depth.estimator import (
        DepthEstimatorProtocol,
        MockDepthEstimator,
        create_depth_estimator,
    )
    from shadowbox.depth.pipeline import DepthPipeline

    # 深度推定器を作成
    if use_mock_depth:
        depth_estimator: DepthEstimatorProtocol = MockDepthEstimator()
    else:
        depth_estimator = create_depth_estimator(settings.depth)

    # 設定ローダーを作成
    config_loader = YAMLConfigLoader(settings.templates_dir)

    return DepthPipeline(
        depth_estimator=depth_estimator,
        config_loader=config_loader,
        depth_to_mesh=depth_to_mesh,
    )
