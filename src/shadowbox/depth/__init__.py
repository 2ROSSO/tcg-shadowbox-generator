"""Depth estimation mode package.

深度推定ベースのシャドーボックス生成パイプラインを提供します。
"""

from shadowbox.depth.estimator import (
    DepthAnythingEstimator,
    DepthEstimatorProtocol,
    MockDepthEstimator,
    create_depth_estimator,
)
from shadowbox.depth.pipeline import DepthPipeline, PipelineResult, ShadowboxPipeline

__all__ = [
    # Pipeline
    "DepthPipeline",
    "PipelineResult",
    "ShadowboxPipeline",  # Deprecated alias for DepthPipeline
    # Estimator protocol and implementations
    "DepthEstimatorProtocol",
    "DepthAnythingEstimator",
    "MockDepthEstimator",
    # Factory
    "create_depth_estimator",
]
