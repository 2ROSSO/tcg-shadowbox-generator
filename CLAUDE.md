# CLAUDE.md - Development Guidelines

## Project Overview

TCG Shadowbox Generator transforms trading card game illustrations into interactive 3D shadowbox displays using AI-powered depth estimation.

## Quick Commands

```bash
# Install dependencies
uv sync

# Install with all optional dependencies (Jupyter + GUI)
uv sync --all-extras

# Run tests
uv run pytest

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/

# Start Jupyter
uv run jupyter lab

# Run GUI app
uv run python -m shadowbox.gui.app
```

## Architecture

- **Dependency Injection**: All core classes accept dependencies via constructor
- **Protocols**: Use `typing.Protocol` for interfaces (enables easy model swapping)
- **Data Classes**: Use `@dataclass(frozen=True)` for immutable configs
- **NumPy Arrays**: All intermediate data uses `numpy.ndarray` for portability
- **Lazy Loading**: Heavy dependencies (torch, transformers) loaded on first use

## Key Modules

| Module | Purpose |
|--------|---------|
| `core/depth.py` | Depth estimation using Depth Anything v2 (swappable) |
| `core/clustering.py` | K-Means layer clustering with optimal k detection |
| `core/mesh.py` | 3D mesh generation from depth layers |
| `core/pipeline.py` | Main orchestrator combining all steps |
| `detection/region.py` | Auto-detect illustration area |
| `gui/template_editor.py` | Manual region selection GUI |
| `gui/image_selector.py` | Image gallery with index selection |
| `visualization/render.py` | Vedo-based 3D renderer |

## Data Flow

```
Image → Crop illustration area → Depth estimation → Clustering → Mesh generation → 3D render
          ↓                          ↓                  ↓              ↓
      template.yaml             depth map         k layers      numpy arrays
```

## Depth Model Swapping

Models implement `DepthEstimatorProtocol`:

```python
class DepthEstimatorProtocol(Protocol):
    def estimate(self, image: Image.Image) -> NDArray[np.float32]: ...
```

Switch models via settings:

```python
settings = ShadowboxSettings()
settings.depth.model_type = "midas"  # or "depth_anything", "zoedepth"
```

## Testing

- Unit tests in `tests/` directory
- Use pytest fixtures from `conftest.py`
- Mock depth estimation model for faster tests
- Run: `uv run pytest -v`

## Code Style

- Python 3.10+ features allowed
- Type hints required for all public functions
- Docstrings in Google style
- Max line length: 100 characters
- Use `ruff` for linting
