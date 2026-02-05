"""スタンドアロンGUIアプリケーション。

このモジュールは、PyQt6を使用したスタンドアロンの
シャドーボックス生成アプリケーションを提供します。

使用方法:
    python -m shadowbox.gui.app
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from shadowbox.config.template import BoundingBox

try:
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class ShadowboxApp(QMainWindow):
    """シャドーボックス生成GUIアプリケーション。

    画像選択、領域設定、パラメータ調整、3D表示を統合した
    スタンドアロンアプリケーション。

    Example:
        >>> app = QApplication(sys.argv)
        >>> window = ShadowboxApp()
        >>> window.show()
        >>> sys.exit(app.exec())
    """

    def __init__(self) -> None:
        super().__init__()

        self._image: Image.Image | None = None
        self._result = None
        self._bbox: BoundingBox | None = None
        self._thread = None

        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle("TCG Shadowbox Generator")
        self.setMinimumSize(1000, 700)

        self._create_menu_bar()

        # Central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Image preview
        from shadowbox.gui.widgets.image_preview import ImagePreview

        self.image_preview = ImagePreview()
        self.image_preview.region_selected.connect(self._on_region_selected)
        self.image_preview.region_cleared.connect(self._on_region_cleared)
        main_layout.addWidget(self.image_preview, stretch=3)

        # Right: Settings + actions
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        from shadowbox.gui.widgets.action_buttons import ActionButtons
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        self.settings_panel = SettingsPanel()
        right_layout.addWidget(self.settings_panel, stretch=1)

        self.action_buttons = ActionButtons()
        self.action_buttons.generate_clicked.connect(self._process_image)
        self.action_buttons.view_3d_clicked.connect(self._show_3d_view)
        self.action_buttons.export_clicked.connect(self._export_mesh)
        right_layout.addWidget(self.action_buttons)

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("画像を読み込んでください")

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

    def _create_menu_bar(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("ファイル")
        open_action = QAction("画像を開く...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("終了", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("ヘルプ")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ---- Image loading ----

    def _open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "画像を開く",
            "",
            "画像ファイル (*.png *.jpg *.jpeg *.gif *.bmp);;すべてのファイル (*.*)",
        )
        if file_path:
            try:
                self._image = Image.open(file_path).convert("RGB")
                self.image_preview.set_image(self._image)
                self.action_buttons.set_has_image(True)
                self._result = None
                self.action_buttons.set_has_result(False)
                self._bbox = None
                self._status_bar.showMessage(f"読み込み完了: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(
                    self, "エラー", f"画像の読み込みに失敗しました:\n{e}"
                )

    # ---- Region selection ----

    def _on_region_selected(self, x: int, y: int, w: int, h: int) -> None:
        from shadowbox.config.template import BoundingBox

        self._bbox = BoundingBox(x=x, y=y, width=w, height=h)
        self._status_bar.showMessage(f"領域選択: ({x}, {y}) {w}x{h}")

    def _on_region_cleared(self) -> None:
        self._bbox = None
        self._status_bar.showMessage("領域選択をクリア")

    # ---- Processing ----

    def _process_image(self) -> None:
        if self._image is None:
            return

        self.action_buttons.generate_btn.setEnabled(False)
        self.action_buttons.set_has_result(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)

        from shadowbox.gui.processing import ProcessingThread

        gs = self.settings_panel.get_gui_settings()
        self._thread = ProcessingThread(
            image=self._image,
            settings=gs,
            bbox=self._bbox,
        )
        self._thread.finished.connect(self._on_processing_finished)
        self._thread.error.connect(self._on_processing_error)
        self._thread.progress.connect(self._status_bar.showMessage)
        self._thread.start()

    def _on_processing_finished(self, result) -> None:
        self._result = result
        self._progress_bar.setVisible(False)
        self.action_buttons.set_has_image(True)
        self.action_buttons.set_has_result(True)
        self._status_bar.showMessage(
            f"処理完了: {result.mesh.num_layers}レイヤー, "
            f"{result.mesh.total_vertices:,}頂点"
        )

        # Update preview tabs
        if hasattr(result, "depth_map"):
            self.image_preview.set_depth_map(result.depth_map)
        if hasattr(result, "labels") and hasattr(result, "cropped_image"):
            self.image_preview.set_labels(result.labels, result.cropped_image)

    def _on_processing_error(self, error_msg: str) -> None:
        self._progress_bar.setVisible(False)
        self.action_buttons.set_has_image(self._image is not None)
        QMessageBox.critical(self, "エラー", f"処理に失敗しました:\n{error_msg}")
        self._status_bar.showMessage("処理エラー")

    # ---- 3D View ----

    def _show_3d_view(self) -> None:
        if self._result is None:
            return
        try:
            from shadowbox.gui.settings_bridge import gui_to_render_options
            from shadowbox.visualization import render_shadowbox

            gs = self.settings_panel.get_gui_settings()
            options = gui_to_render_options(gs)
            render_shadowbox(self._result.mesh, options)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"3D表示に失敗しました:\n{e}")

    # ---- Export ----

    def _export_mesh(self) -> None:
        if self._result is None:
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "エクスポート",
            "shadowbox",
            "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)",
        )
        if not file_path:
            return

        try:
            from shadowbox.visualization.export import (
                export_to_obj,
                export_to_ply,
                export_to_stl,
            )

            if file_path.endswith(".obj") or "OBJ" in selected_filter:
                if not file_path.endswith(".obj"):
                    file_path += ".obj"
                export_to_obj(self._result.mesh, file_path)
            elif file_path.endswith(".ply") or "PLY" in selected_filter:
                if not file_path.endswith(".ply"):
                    file_path += ".ply"
                export_to_ply(self._result.mesh, file_path)
            else:
                if not file_path.endswith(".stl"):
                    file_path += ".stl"
                export_to_stl(self._result.mesh, file_path)

            self._status_bar.showMessage(f"エクスポート完了: {file_path}")
        except Exception as e:
            QMessageBox.critical(
                self, "エラー", f"エクスポートに失敗しました:\n{e}"
            )

    # ---- About ----

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About TCG Shadowbox Generator",
            "TCG Shadowbox Generator\n\n"
            "TCGカードのイラストを深度推定とクラスタリングで階層化し、\n"
            "インタラクティブな3Dシャドーボックスとして表示するツール。\n\n"
            "https://github.com/2ROSSO/shadowbox-generator",
        )


def main() -> None:
    """アプリケーションのエントリーポイント。"""
    if not PYQT_AVAILABLE:
        print("エラー: PyQt6がインストールされていません。")
        print("インストール方法: uv pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)

    from shadowbox.gui.theme import DARK_STYLESHEET

    app.setStyleSheet(DARK_STYLESHEET)

    window = ShadowboxApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
