"""3D表示設定タブ。"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class RenderingTab(QWidget):
    """3D表示設定タブ。

    Signals:
        settings_changed: 設定が変更されたとき。
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg_color = (30, 30, 30)
        self._init_ui()
        self._connect_signals()
        # デフォルトを mesh に設定（シグナル接続後に呼び表示切替を発火）
        self.render_mode.setCurrentText("mesh")

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Render Mode
        row = QHBoxLayout()
        row.addWidget(QLabel("描画モード:"))
        self.render_mode = QComboBox()
        self.render_mode.addItems(["points", "mesh"])
        row.addWidget(self.render_mode)
        layout.addLayout(row)

        # Point Size
        self._point_row = QHBoxLayout()
        self._point_row_label = QLabel("ポイントサイズ:")
        self._point_row.addWidget(self._point_row_label)
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(3)
        self._point_row.addWidget(self.point_size_slider)
        self.point_size_label = QLabel("3")
        self.point_size_label.setFixedWidth(20)
        self._point_row.addWidget(self.point_size_label)
        layout.addLayout(self._point_row)

        # Mesh Size
        row = QHBoxLayout()
        self._mesh_label = QLabel("メッシュサイズ:")
        row.addWidget(self._mesh_label)
        self.mesh_size = QDoubleSpinBox()
        self.mesh_size.setRange(0.001, 0.05)
        self.mesh_size.setSingleStep(0.001)
        self.mesh_size.setDecimals(3)
        self.mesh_size.setValue(0.008)
        row.addWidget(self.mesh_size)
        layout.addLayout(row)
        self._mesh_label.setVisible(False)
        self.mesh_size.setVisible(False)

        # Show Axes
        self.show_axes = QCheckBox("軸を表示")
        layout.addWidget(self.show_axes)

        # Show Frame
        self.show_frame = QCheckBox("フレームを表示")
        self.show_frame.setChecked(True)
        layout.addWidget(self.show_frame)

        # Layer Opacity
        row = QHBoxLayout()
        row.addWidget(QLabel("不透明度:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        row.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(40)
        row.addWidget(self.opacity_label)
        layout.addLayout(row)

        # Background Color
        row = QHBoxLayout()
        row.addWidget(QLabel("背景色:"))
        self.bg_color_btn = QPushButton()
        self._update_color_button()
        self.bg_color_btn.clicked.connect(self._pick_color)
        row.addWidget(self.bg_color_btn)
        layout.addLayout(row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.render_mode.currentTextChanged.connect(self._on_render_mode_changed)
        self.point_size_slider.valueChanged.connect(
            lambda v: self.point_size_label.setText(str(v))
        )
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%")
        )

        self.render_mode.currentIndexChanged.connect(self.settings_changed)
        self.point_size_slider.valueChanged.connect(self.settings_changed)
        self.mesh_size.valueChanged.connect(self.settings_changed)
        self.show_axes.toggled.connect(self.settings_changed)
        self.show_frame.toggled.connect(self.settings_changed)
        self.opacity_slider.valueChanged.connect(self.settings_changed)

    def _on_render_mode_changed(self, mode: str) -> None:
        is_points = mode == "points"
        self._point_row_label.setVisible(is_points)
        self.point_size_slider.setVisible(is_points)
        self.point_size_label.setVisible(is_points)
        self._mesh_label.setVisible(not is_points)
        self.mesh_size.setVisible(not is_points)

    def _pick_color(self) -> None:
        r, g, b = self._bg_color
        color = QColorDialog.getColor(QColor(r, g, b), self, "背景色を選択")
        if color.isValid():
            self._bg_color = (color.red(), color.green(), color.blue())
            self._update_color_button()
            self.settings_changed.emit()

    def _update_color_button(self) -> None:
        r, g, b = self._bg_color
        self.bg_color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); "
            f"border: 1px solid #666; min-width: 60px; min-height: 20px;"
        )
        self.bg_color_btn.setText(f"({r},{g},{b})")

    def set_values(self, values: dict) -> None:
        """辞書から設定値を復元。"""
        if "render_mode" in values:
            self.render_mode.setCurrentText(values["render_mode"])
        if "point_size" in values:
            self.point_size_slider.setValue(int(values["point_size"]))
        if "mesh_size" in values:
            self.mesh_size.setValue(values["mesh_size"])
        if "show_axes" in values:
            self.show_axes.setChecked(values["show_axes"])
        if "show_frame_3d" in values:
            self.show_frame.setChecked(values["show_frame_3d"])
        if "layer_opacity" in values:
            self.opacity_slider.setValue(int(values["layer_opacity"] * 100))
        if "background_color" in values:
            self._bg_color = tuple(values["background_color"])
            self._update_color_button()

    def get_values(self) -> dict:
        """現在の設定値を辞書で返す。"""
        return {
            "render_mode": self.render_mode.currentText(),
            "point_size": float(self.point_size_slider.value()),
            "mesh_size": self.mesh_size.value(),
            "show_axes": self.show_axes.isChecked(),
            "show_frame_3d": self.show_frame.isChecked(),
            "layer_opacity": self.opacity_slider.value() / 100.0,
            "background_color": self._bg_color,
        }
