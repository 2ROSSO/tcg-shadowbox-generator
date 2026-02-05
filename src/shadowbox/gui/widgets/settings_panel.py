"""設定パネル（4タブコンテナ）。"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from shadowbox.gui.settings_bridge import GuiSettings
from shadowbox.gui.widgets.frame_tab import FrameTab
from shadowbox.gui.widgets.layers_tab import LayersTab
from shadowbox.gui.widgets.processing_tab import ProcessingTab
from shadowbox.gui.widgets.rendering_tab import RenderingTab


class SettingsPanel(QWidget):
    """4タブの設定パネル。

    Signals:
        settings_changed: いずれかのタブ設定が変更されたとき。
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.processing_tab = ProcessingTab()
        self.layers_tab = LayersTab()
        self.frame_tab = FrameTab()
        self.rendering_tab = RenderingTab()

        self.tabs.addTab(self.processing_tab, "処理")
        self.tabs.addTab(self.layers_tab, "レイヤー")
        self.tabs.addTab(self.frame_tab, "フレーム")
        self.tabs.addTab(self.rendering_tab, "3D表示")

        layout.addWidget(self.tabs)

        # Forward settings_changed from all tabs
        self.processing_tab.settings_changed.connect(self.settings_changed)
        self.layers_tab.settings_changed.connect(self.settings_changed)
        self.frame_tab.settings_changed.connect(self.settings_changed)
        self.rendering_tab.settings_changed.connect(self.settings_changed)

    def get_gui_settings(self) -> GuiSettings:
        """全タブから GuiSettings を構築。"""
        values: dict = {}
        values.update(self.processing_tab.get_values())
        values.update(self.layers_tab.get_values())
        values.update(self.frame_tab.get_values())
        values.update(self.rendering_tab.get_values())
        return GuiSettings(**values)
