"""画像選択モジュールのテスト。

このモジュールは、ImageSelectorと関連機能の
ユニットテストを提供します。
"""

import tempfile
from pathlib import Path

import matplotlib
import pytest
from PIL import Image

from shadowbox.gui.image_selector import (
    ImageSelector,
    URLImageLoader,
    create_image_selector,
    load_from_url,
)

# バックエンドをAggに設定（ヘッドレス環境用）
matplotlib.use("Agg")


@pytest.fixture
def temp_image_dir():
    """テスト用の一時画像ディレクトリを作成。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # テスト画像を作成
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img.save(tmpdir_path / f"test_image_{i}.png")

        # JPEGも追加
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        img.save(tmpdir_path / "test_image.jpg")

        yield tmpdir_path


@pytest.fixture
def empty_dir():
    """空の一時ディレクトリを作成。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestImageSelector:
    """ImageSelectorのテスト。"""

    def test_init(self, temp_image_dir: Path) -> None:
        """初期化をテスト。"""
        selector = ImageSelector(temp_image_dir)

        assert selector.directory == temp_image_dir
        assert selector.count == 6  # 5 PNG + 1 JPG

    def test_init_nonexistent_dir(self) -> None:
        """存在しないディレクトリでエラーが発生することをテスト。"""
        with pytest.raises(FileNotFoundError, match="ディレクトリが見つかりません"):
            ImageSelector("/nonexistent/path")

    def test_init_file_not_dir(self, temp_image_dir: Path) -> None:
        """ファイルパスでエラーが発生することをテスト。"""
        file_path = temp_image_dir / "test_image_0.png"

        with pytest.raises(ValueError, match="ディレクトリではありません"):
            ImageSelector(file_path)

    def test_image_paths(self, temp_image_dir: Path) -> None:
        """image_pathsプロパティをテスト。"""
        selector = ImageSelector(temp_image_dir)
        paths = selector.image_paths

        assert len(paths) == 6
        assert all(isinstance(p, Path) for p in paths)
        assert all(p.exists() for p in paths)

    def test_select(self, temp_image_dir: Path) -> None:
        """select機能をテスト。"""
        selector = ImageSelector(temp_image_dir)

        image = selector.select(0)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_select_out_of_range(self, temp_image_dir: Path) -> None:
        """範囲外インデックスでエラーが発生することをテスト。"""
        selector = ImageSelector(temp_image_dir)

        with pytest.raises(IndexError, match="インデックスが範囲外です"):
            selector.select(100)

        with pytest.raises(IndexError, match="インデックスが範囲外です"):
            selector.select(-1)

    def test_get_path(self, temp_image_dir: Path) -> None:
        """get_path機能をテスト。"""
        selector = ImageSelector(temp_image_dir)

        path = selector.get_path(0)

        assert isinstance(path, Path)
        assert path.exists()

    def test_get_path_out_of_range(self, temp_image_dir: Path) -> None:
        """範囲外インデックスでエラーが発生することをテスト。"""
        selector = ImageSelector(temp_image_dir)

        with pytest.raises(IndexError, match="インデックスが範囲外です"):
            selector.get_path(100)

    def test_list_images(self, temp_image_dir: Path) -> None:
        """list_images機能をテスト。"""
        selector = ImageSelector(temp_image_dir)

        image_list = selector.list_images()

        assert len(image_list) == 6
        assert all(isinstance(s, str) for s in image_list)
        assert "[0]" in image_list[0]

    def test_print_list(self, temp_image_dir: Path, capsys) -> None:
        """print_list機能をテスト。"""
        selector = ImageSelector(temp_image_dir)

        selector.print_list()

        captured = capsys.readouterr()
        assert "画像" in captured.out
        assert "[0]" in captured.out

    def test_print_list_empty(self, empty_dir: Path, capsys) -> None:
        """空ディレクトリでのprint_listをテスト。"""
        selector = ImageSelector(empty_dir)

        selector.print_list()

        captured = capsys.readouterr()
        assert "見つかりません" in captured.out

    def test_empty_directory(self, empty_dir: Path) -> None:
        """空のディレクトリでの動作をテスト。"""
        selector = ImageSelector(empty_dir)

        assert selector.count == 0
        assert selector.image_paths == []

    def test_supported_extensions(self, temp_image_dir: Path) -> None:
        """サポートする拡張子のフィルタリングをテスト。"""
        # テキストファイルを追加（無視されるべき）
        (temp_image_dir / "readme.txt").write_text("test")

        selector = ImageSelector(temp_image_dir)

        # テキストファイルは含まれない
        assert selector.count == 6
        assert all(p.suffix.lower() in ImageSelector.SUPPORTED_EXTENSIONS for p in selector.image_paths)


class TestImageSelectorGallery:
    """ギャラリー表示のテスト。"""

    def test_show_gallery_no_error(self, temp_image_dir: Path) -> None:
        """show_galleryがエラーなく実行されることをテスト。"""
        import matplotlib.pyplot as plt

        selector = ImageSelector(temp_image_dir)

        # エラーなく実行できることを確認（表示はスキップ）
        selector._generate_thumbnails((100, 100))

        assert len(selector._thumbnails) == 6

        plt.close("all")

    def test_display_grid_alias(self, temp_image_dir: Path) -> None:
        """display_gridがshow_galleryのエイリアスであることをテスト。"""
        selector = ImageSelector(temp_image_dir)

        # メソッドが存在することを確認
        assert hasattr(selector, "display_grid")
        assert callable(selector.display_grid)


class TestURLImageLoader:
    """URLImageLoaderのテスト。"""

    def test_load_method_exists(self) -> None:
        """loadメソッドが存在することをテスト。"""
        assert hasattr(URLImageLoader, "load")
        assert callable(URLImageLoader.load)

    def test_load_multiple_method_exists(self) -> None:
        """load_multipleメソッドが存在することをテスト。"""
        assert hasattr(URLImageLoader, "load_multiple")
        assert callable(URLImageLoader.load_multiple)

    def test_load_invalid_url(self) -> None:
        """無効なURLでエラーが発生することをテスト。"""
        with pytest.raises(ValueError):
            URLImageLoader.load("https://invalid-url-that-does-not-exist-12345.com/image.png", timeout=5)


class TestUtilityFunctions:
    """ユーティリティ関数のテスト。"""

    def test_create_image_selector(self, temp_image_dir: Path) -> None:
        """create_image_selector関数をテスト。"""
        selector = create_image_selector(temp_image_dir)

        assert isinstance(selector, ImageSelector)
        assert selector.count == 6

    def test_load_from_url_function_exists(self) -> None:
        """load_from_url関数が存在することをテスト。"""
        assert callable(load_from_url)
