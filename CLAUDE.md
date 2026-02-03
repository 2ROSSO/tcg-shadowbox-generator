# CLAUDE.md - Development Guidelines

> **重要**: 実装を開始する前に、必ずこのファイル全体を読んでください。

## Project Overview

TCG Shadowbox Generator transforms trading card game illustrations into interactive 3D shadowbox displays using AI-powered depth estimation.

## Quick Commands

```bash
# Install dependencies
uv sync

# Install with all optional dependencies (Jupyter + GUI + TripoSR)
uv sync --all-extras

# Run tests (required before commit)
uv run pytest

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/

# Start Jupyter
uv run jupyter lab

# Run GUI app
uv run python -m shadowbox.gui.app

# Clear notebook outputs (required before commit)
uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Architecture

### Design Principles

- **Dependency Injection**: All core classes accept dependencies via constructor
- **Protocols**: Use `typing.Protocol` for interfaces (enables easy model swapping)
- **Data Classes**: Use `@dataclass(frozen=True)` for immutable configs
- **NumPy Arrays**: All intermediate data uses `numpy.ndarray` for portability
- **Lazy Loading**: Heavy dependencies (torch, transformers) loaded on first use
- **Factory Pattern**: Use `create_*` functions to instantiate configured objects

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      create_pipeline()                          │
│  settings.model_mode で分岐:                                     │
│    - "depth"   → ShadowboxPipeline (深度推定+クラスタリング)      │
│    - "triposr" → TripoSRPipeline (直接3Dメッシュ生成)            │
└─────────────────────────────────────────────────────────────────┘
```

### Dependency Injection Example

```python
# ShadowboxPipeline はコンストラクタで全依存を受け取る
class ShadowboxPipeline:
    def __init__(
        self,
        depth_estimator: DepthEstimatorProtocol,  # Protocol で型定義
        clusterer: LayerClustererProtocol,
        mesh_generator: MeshGenerator,
        config_loader: ConfigLoaderProtocol,
    ) -> None:
        ...

# create_pipeline() が依存関係を組み立てる
def create_pipeline(settings: ShadowboxSettings) -> ShadowboxPipeline:
    depth_estimator = create_depth_estimator(settings.depth)
    clusterer = KMeansLayerClusterer(settings.clustering)
    ...
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `core/depth.py` | Depth estimation using Depth Anything v2 (swappable via Protocol) |
| `core/clustering.py` | K-Means layer clustering with optimal k detection |
| `core/mesh.py` | 3D mesh generation from depth layers |
| `core/pipeline.py` | Main orchestrator, `create_pipeline()` factory |
| `triposr/generator.py` | TripoSR 3D mesh generation (alternative to depth+clustering) |
| `triposr/pipeline.py` | TripoSR pipeline wrapper |
| `detection/region.py` | Auto-detect illustration area |
| `gui/template_editor.py` | Manual region selection GUI |
| `gui/image_selector.py` | Image gallery with index selection |
| `visualization/render.py` | Vedo-based 3D renderer |
| `visualization/export.py` | STL/OBJ/PLY export |

## Data Flow

### Depth Mode (default)
```
Image → Crop → Depth estimation → Clustering → Mesh generation → 3D render
          ↓           ↓                ↓              ↓
      BoundingBox  depth map       k layers      ShadowboxMesh
```

### TripoSR Mode
```
Image → Crop → TripoSR model → 3D mesh → ShadowboxMesh → 3D render
          ↓                       ↓
      BoundingBox            trimesh object
```

## Protocol Interfaces

### DepthEstimatorProtocol
```python
class DepthEstimatorProtocol(Protocol):
    def estimate(self, image: Image.Image) -> NDArray[np.float32]: ...
```

### LayerClustererProtocol
```python
class LayerClustererProtocol(Protocol):
    def cluster(self, depth_map: NDArray, k: int) -> tuple[NDArray, NDArray]: ...
    def find_optimal_k(self, depth_map: NDArray) -> int: ...
```

### ConfigLoaderProtocol
```python
class ConfigLoaderProtocol(Protocol):
    def load_template(self, name: str) -> CardTemplate: ...
    def save_template(self, template: CardTemplate) -> None: ...
```

## Settings Structure

```python
ShadowboxSettings
├── model_mode: str          # "depth" or "triposr"
├── depth: DepthSettings     # Depth model config
├── clustering: ClusteringSettings
├── render: RenderSettings   # Layer options, frame settings
├── triposr: TripoSRSettings # TripoSR-specific config
└── templates_dir: Path
```

## Testing

### Test Structure
- Unit tests in `tests/` directory
- Use pytest fixtures from `conftest.py`
- Mock depth estimation with `MockDepthEstimator` for faster tests
- Use `@pytest.mark.slow` / `@pytest.mark.integration` for heavy tests

### Running Tests
```bash
# Standard tests (excludes slow/integration)
uv run pytest

# All tests including slow ones
uv run pytest -m ""

# Specific module
uv run pytest tests/test_triposr.py -v
```

### Test Requirements
- 新しいモジュールには必ずテストを作成
- Protocol を実装するクラスは、そのインターフェースのテストを含める
- `create_pipeline(use_mock_depth=True)` でモック深度推定器を使用

## Code Style

- Python 3.10+ features allowed
- Type hints required for all public functions
- Docstrings in Google style (日本語可)
- Max line length: 100 characters
- Use `ruff` for linting

## Notebook

### 01_quickstart.ipynb
- ALL RUN で一気に実行できるクイックスタート
- 設定変数は先頭セルで集中管理
- `MODEL_MODE`: "depth" or "triposr" で切り替え

### 注意事項
- **コミット前に出力をクリア**: `uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- 大きな出力（画像、3Dレンダリング結果）はgitに含めない

## TripoSR Integration

### Setup
```bash
# 1. Clone TripoSR (プロジェクトルートに配置)
git clone https://github.com/VAST-AI-Research/TripoSR.git

# 2. Install torchmcubes (C++拡張、Visual Studio Build Tools必要)
uv pip install scikit-build-core cmake ninja pybind11
set CMAKE_PREFIX_PATH=.venv\Lib\site-packages\torch\share\cmake
uv pip install git+https://github.com/tatsy/torchmcubes.git --no-build-isolation
```

### Path Auto-Detection
`TripoSRGenerator` は以下の場所から自動的に TripoSR を検出:
1. カレントディレクトリの `TripoSR/`
2. プロジェクトルート（pyproject.toml がある場所）の `TripoSR/`

### 注意
- `rembg` は Python 3.11+ と依存関係競合があるため、オプション化済み
- `/TripoSR/` は `.gitignore` でルートのみ除外（`src/shadowbox/triposr` は含める）

---

## Commit Checklist

コミット前に必ず以下を確認:

- [ ] `uv run pytest` が成功すること
- [ ] `uv run ruff check src/` でエラーがないこと
- [ ] Notebook の出力がクリアされていること
- [ ] 新機能には対応するテストがあること
- [ ] DI パターンに従っていること（直接インスタンス化ではなくファクトリー経由）
- [ ] Protocol を使用してインターフェースを定義していること

## Implementation Guidelines

### 新機能を追加する際

1. **CLAUDE.md を読む** - アーキテクチャと既存パターンを理解
2. **Protocol を定義** - 新しいインターフェースが必要な場合
3. **DI を考慮** - コンストラクタで依存を受け取る設計
4. **Settings に追加** - 設定可能なパラメータは `ShadowboxSettings` に集約
5. **Factory を更新** - `create_pipeline()` や専用ファクトリーを更新
6. **テストを作成** - ユニットテスト必須、統合テストは `@pytest.mark.integration`
7. **Notebook で動作確認** - 実際のユースケースで確認
8. **Commit Checklist を実行**

### 既存機能を変更する際

1. **影響範囲を確認** - Protocol を実装しているクラスを `Grep` で検索
2. **テストを先に修正** - 期待する動作をテストで定義
3. **実装を変更**
4. **全テストが通ることを確認**
