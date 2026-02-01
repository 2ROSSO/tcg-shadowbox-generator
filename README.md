# TCG Shadowbox Generator

Transform TCG card illustrations into interactive 3D shadowbox displays using AI-powered depth estimation.

## Features

- **Depth Estimation**: Uses Depth Anything v2 (lightweight, Apache 2.0 license)
- **Auto/Manual Region Selection**: Automatically detect or manually select illustration areas
- **Layer Clustering**: Automatically finds optimal number of layers using silhouette analysis
- **Interactive 3D View**: Rotate and explore shadowbox with mouse
- **Multiple Input Methods**: Load from URL or select from local directory gallery
- **Template System**: Save and reuse card templates (Pokemon, MTG, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/2ROSSO/shadowbox-generator.git
cd shadowbox-generator

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Optional Dependencies

```bash
# For Jupyter notebook support
uv sync --extra jupyter

# For standalone GUI app (PyQt6)
uv sync --extra gui

# For everything
uv sync --all-extras
```

| オプション | 含まれるパッケージ                 | 用途                                                      |
| ---------- | ---------------------------------- | --------------------------------------------------------- |
| `jupyter`  | jupyter, ipykernel, ipympl, plotly | Jupyter Notebook での実行、**手動領域選択**、**3Dビュー** |
| `gui`      | PyQt6                              | スタンドアロンGUIアプリ                                   |
| `all`      | 上記すべて                         | フル機能                                                  |

> **Note**: Jupyter Notebook で以下の機能を使用するには `jupyter` オプションが必要です:
>
> - **手動領域選択**: `ipympl` により `%matplotlib widget` バックエンドでインタラクティブな操作が可能
> - **3Dビュー**: `plotly` により `render_shadowbox()` がセル内でインタラクティブ3D表示されます

## Quick Start

### Using Jupyter Notebook

```bash
uv run jupyter lab
# Open notebooks/01_quickstart.ipynb
```

### Using Python

```python
from PIL import Image
from shadowbox import create_pipeline, ShadowboxSettings

# Load image
image = Image.open("card.png")

# Create pipeline with default settings
pipeline = create_pipeline()

# Process image (auto-detect illustration area)
result = pipeline.process(image, auto_detect=True)

# View 3D shadowbox
from shadowbox.visualization import ShadowboxRenderer
renderer = ShadowboxRenderer(ShadowboxSettings().render)
renderer.render(result.mesh)
```

### Using GUI App

```bash
uv run python -m shadowbox.gui.app
```

## How It Works

1. **Load Image**: From URL or local directory
2. **Select Region**: Auto-detect or manually draw illustration area
3. **Depth Estimation**: AI model estimates per-pixel depth
4. **Clustering**: Depth values clustered into discrete layers
5. **3D Generation**: Layers rendered as stacked 3D surfaces
6. **Interactive View**: Explore with mouse rotation/zoom

## Project Structure

```text
shadowbox-generator/
├── src/shadowbox/
│   ├── core/           # Pipeline, depth, clustering, mesh
│   ├── config/         # Settings and templates
│   ├── visualization/  # 2D/3D rendering
│   ├── detection/      # Auto region detection
│   └── gui/            # GUI components
├── notebooks/          # Jupyter notebooks
├── data/templates/     # Card templates (YAML)
└── tests/              # Test suite
```

## Swapping Depth Models

The depth estimator is pluggable. Currently supported:

- **Depth Anything v2** (default): Apache 2.0, lightweight, high accuracy
- **MiDaS**: MIT license, mature ecosystem
- **ZoeDepth**: MIT license, metric depth support

```python
from shadowbox.config import ShadowboxSettings

settings = ShadowboxSettings()
settings.depth.model_type = "midas"  # Switch model
pipeline = create_pipeline(settings)
```

## Development

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/
```

## License

MIT License
