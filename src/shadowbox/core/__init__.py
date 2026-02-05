"""Core business logic modules.

共通コンポーネントを提供します。
"""

from shadowbox.core.back_panel_factory import create_back_panel
from shadowbox.core.frame_factory import (
    FrameConfig,
    calculate_bounds,
    create_frame,
    create_plane_frame,
    create_walled_frame,
)
from shadowbox.core.mesh import MeshGeneratorProtocol
from shadowbox.core.pipeline import BasePipelineResult

__all__ = [
    # Pipeline base
    "BasePipelineResult",
    # Frame
    "FrameConfig",
    "calculate_bounds",
    "create_back_panel",
    "create_frame",
    "create_plane_frame",
    "create_walled_frame",
    # Mesh
    "MeshGeneratorProtocol",
]
