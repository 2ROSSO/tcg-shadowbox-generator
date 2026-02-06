"""設定パネル（4タブコンテナ）。"""

from __future__ import annotations

from dataclasses import asdict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from shadowbox.gui.i18n import tr
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

        self.tabs.addTab(self.processing_tab, tr("settings.processing"))
        self.tabs.addTab(self.layers_tab, tr("settings.layers"))
        self.tabs.addTab(self.frame_tab, tr("settings.frame"))
        self.tabs.addTab(self.rendering_tab, tr("settings.rendering"))

        layout.addWidget(self.tabs)

        # Forward settings_changed from all tabs
        self.processing_tab.settings_changed.connect(self.settings_changed)
        self.layers_tab.settings_changed.connect(self.settings_changed)
        self.frame_tab.settings_changed.connect(self.settings_changed)
        self.rendering_tab.settings_changed.connect(self.settings_changed)

    def retranslate(self) -> None:
        """言語変更時にUI文字列を更新。"""
        self.tabs.setTabText(0, tr("settings.processing"))
        self.tabs.setTabText(1, tr("settings.layers"))
        self.tabs.setTabText(2, tr("settings.frame"))
        self.tabs.setTabText(3, tr("settings.rendering"))
        self.processing_tab.retranslate()
        self.layers_tab.retranslate()
        self.frame_tab.retranslate()
        self.rendering_tab.retranslate()

    def set_gui_settings(self, gs: GuiSettings) -> None:
        """GuiSettings の値を全タブに配布。"""
        values = asdict(gs)
        self.processing_tab.set_values(values)
        self.layers_tab.set_values(values)
        self.frame_tab.set_values(values)
        self.rendering_tab.set_values(values)

    def get_gui_settings(self) -> GuiSettings:
        """全タブから GuiSettings を構築。"""
        values: dict = {}
        values.update(self.processing_tab.get_values())
        values.update(self.layers_tab.get_values())
        values.update(self.frame_tab.get_values())
        values.update(self.rendering_tab.get_values())
        return GuiSettings(**values)
