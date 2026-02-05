"""処理設定タブ。"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ProcessingTab(QWidget):
    """処理パラメータ設定タブ。

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

        # Model Mode
        row = QHBoxLayout()
        row.addWidget(QLabel("モデルモード:"))
        self.model_mode = QComboBox()
        self.model_mode.addItems(["depth", "triposr"])
        row.addWidget(self.model_mode)
        layout.addLayout(row)

        # Detection Method
        row = QHBoxLayout()
        row.addWidget(QLabel("検出方法:"))
        self.detection_method = QComboBox()
        self.detection_method.addItems(["auto", "none"])
        row.addWidget(self.detection_method)
        layout.addLayout(row)

        # Mock Depth
        self.mock_depth = QCheckBox("モック深度推定（テスト用）")
        layout.addWidget(self.mock_depth)

        # Raw Depth
        self.raw_depth = QCheckBox("生深度モード")
        layout.addWidget(self.raw_depth)

        # Depth Scale
        row = QHBoxLayout()
        row.addWidget(QLabel("深度スケール:"))
        self.depth_scale = QDoubleSpinBox()
        self.depth_scale.setRange(0.1, 5.0)
        self.depth_scale.setSingleStep(0.1)
        self.depth_scale.setValue(1.0)
        self.depth_scale.setEnabled(False)
        row.addWidget(self.depth_scale)
        layout.addLayout(row)

        # Num Layers
        row = QHBoxLayout()
        row.addWidget(QLabel("レイヤー数:"))
        self.num_layers_auto = QCheckBox("自動")
        self.num_layers_auto.setChecked(True)
        row.addWidget(self.num_layers_auto)
        self.num_layers = QSpinBox()
        self.num_layers.setRange(1, 10)
        self.num_layers.setValue(5)
        self.num_layers.setEnabled(False)
        row.addWidget(self.num_layers)
        layout.addLayout(row)

        # Max Resolution
        row = QHBoxLayout()
        row.addWidget(QLabel("最大解像度:"))
        self.max_res_unlimited = QCheckBox("無制限")
        self.max_res_unlimited.setChecked(True)
        row.addWidget(self.max_res_unlimited)
        self.max_resolution = QSpinBox()
        self.max_resolution.setRange(50, 2000)
        self.max_resolution.setValue(500)
        self.max_resolution.setEnabled(False)
        row.addWidget(self.max_resolution)
        layout.addLayout(row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.raw_depth.toggled.connect(self._on_raw_depth_toggled)
        self.num_layers_auto.toggled.connect(self._on_auto_layers_toggled)
        self.max_res_unlimited.toggled.connect(self._on_unlimited_res_toggled)

        # Emit settings_changed for all controls
        self.model_mode.currentIndexChanged.connect(self.settings_changed)
        self.detection_method.currentIndexChanged.connect(self.settings_changed)
        self.mock_depth.toggled.connect(self.settings_changed)
        self.raw_depth.toggled.connect(self.settings_changed)
        self.depth_scale.valueChanged.connect(self.settings_changed)
        self.num_layers_auto.toggled.connect(self.settings_changed)
        self.num_layers.valueChanged.connect(self.settings_changed)
        self.max_res_unlimited.toggled.connect(self.settings_changed)
        self.max_resolution.valueChanged.connect(self.settings_changed)

    def _on_raw_depth_toggled(self, checked: bool) -> None:
        self.depth_scale.setEnabled(checked)
        self.num_layers.setEnabled(not checked and not self.num_layers_auto.isChecked())
        self.num_layers_auto.setEnabled(not checked)

    def _on_auto_layers_toggled(self, checked: bool) -> None:
        self.num_layers.setEnabled(not checked and not self.raw_depth.isChecked())

    def _on_unlimited_res_toggled(self, checked: bool) -> None:
        self.max_resolution.setEnabled(not checked)

    def get_values(self) -> dict:
        """現在の設定値を辞書で返す。"""
        return {
            "model_mode": self.model_mode.currentText(),
            "detection_method": self.detection_method.currentText(),
            "use_mock_depth": self.mock_depth.isChecked(),
            "use_raw_depth": self.raw_depth.isChecked(),
            "depth_scale": self.depth_scale.value(),
            "num_layers": None if self.num_layers_auto.isChecked() else self.num_layers.value(),
            "max_resolution": (
                None if self.max_res_unlimited.isChecked() else self.max_resolution.value()
            ),
        }
