"""フレーム設定タブ。"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class FrameTab(QWidget):
    """フレーム設定タブ。

    Signals:
        settings_changed: 設定が変更されたとき。
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Include Frame
        self.include_frame = QCheckBox("フレームを含める")
        self.include_frame.setChecked(True)
        layout.addWidget(self.include_frame)

        # Include Card Frame
        self.include_card_frame = QCheckBox("カードフレームを含める")
        self.include_card_frame.setChecked(True)
        layout.addWidget(self.include_card_frame)

        # Frame Depth
        row = QHBoxLayout()
        row.addWidget(QLabel("フレーム厚み:"))
        self.frame_depth = QDoubleSpinBox()
        self.frame_depth.setRange(0.1, 2.0)
        self.frame_depth.setSingleStep(0.1)
        self.frame_depth.setValue(0.5)
        row.addWidget(self.frame_depth)
        layout.addLayout(row)

        # Frame Wall Mode
        row = QHBoxLayout()
        row.addWidget(QLabel("壁モード:"))
        self.frame_wall_mode = QComboBox()
        self.frame_wall_mode.addItems(["none", "outer"])
        self.frame_wall_mode.setCurrentText("outer")
        row.addWidget(self.frame_wall_mode)
        layout.addLayout(row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.include_frame.toggled.connect(self._on_frame_toggled)

        self.include_frame.toggled.connect(self.settings_changed)
        self.include_card_frame.toggled.connect(self.settings_changed)
        self.frame_depth.valueChanged.connect(self.settings_changed)
        self.frame_wall_mode.currentIndexChanged.connect(self.settings_changed)

    def _on_frame_toggled(self, checked: bool) -> None:
        self.frame_depth.setEnabled(checked)
        self.frame_wall_mode.setEnabled(checked)

    def set_values(self, values: dict) -> None:
        """辞書から設定値を復元。"""
        if "include_frame" in values:
            self.include_frame.setChecked(values["include_frame"])
        if "include_card_frame" in values:
            self.include_card_frame.setChecked(values["include_card_frame"])
        if "frame_depth" in values:
            self.frame_depth.setValue(values["frame_depth"])
        if "frame_wall_mode" in values:
            self.frame_wall_mode.setCurrentText(values["frame_wall_mode"])

    def get_values(self) -> dict:
        """現在の設定値を辞書で返す。"""
        return {
            "include_frame": self.include_frame.isChecked(),
            "include_card_frame": self.include_card_frame.isChecked(),
            "frame_depth": self.frame_depth.value(),
            "frame_wall_mode": self.frame_wall_mode.currentText(),
        }
