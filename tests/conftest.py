"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    # Create a simple gradient image
    width, height = 100, 150
    data = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient from red to blue
    for y in range(height):
        for x in range(width):
            data[y, x] = [
                255 - int(255 * y / height),  # R
                100,                           # G
                int(255 * y / height),         # B
            ]

    return Image.fromarray(data, mode="RGB")


@pytest.fixture
def sample_depth_map() -> np.ndarray:
    """Create a sample depth map for testing."""
    height, width = 100, 150
    # Create depth with clear regions
    depth = np.zeros((height, width), dtype=np.float32)

    # Background (far)
    depth[:] = 0.8

    # Middle layer
    depth[20:80, 30:120] = 0.5

    # Foreground (close)
    depth[40:60, 50:100] = 0.2

    return depth


@pytest.fixture
def sample_image_array(sample_image: Image.Image) -> np.ndarray:
    """Convert sample image to numpy array."""
    return np.array(sample_image)
