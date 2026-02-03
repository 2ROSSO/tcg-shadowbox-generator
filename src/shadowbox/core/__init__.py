"""Core business logic modules."""

from shadowbox.core.frame_factory import (
    FrameConfig,
    calculate_bounds,
    create_frame,
    create_plane_frame,
    create_walled_frame,
)

__all__ = [
    "FrameConfig",
    "calculate_bounds",
    "create_frame",
    "create_plane_frame",
    "create_walled_frame",
]
