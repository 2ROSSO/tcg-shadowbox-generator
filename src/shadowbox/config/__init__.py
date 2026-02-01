"""Configuration management modules."""

from shadowbox.config.loader import ConfigLoaderProtocol, YAMLConfigLoader
from shadowbox.config.settings import (
    ClusteringSettings,
    DepthEstimationSettings,
    RenderSettings,
    ShadowboxSettings,
)
from shadowbox.config.template import BoundingBox, CardTemplate

__all__ = [
    "ShadowboxSettings",
    "DepthEstimationSettings",
    "ClusteringSettings",
    "RenderSettings",
    "CardTemplate",
    "BoundingBox",
    "ConfigLoaderProtocol",
    "YAMLConfigLoader",
]
