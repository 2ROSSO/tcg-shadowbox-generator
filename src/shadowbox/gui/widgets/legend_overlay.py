"""凡例オーバーレイウィジェット。

レイヤータブや深度タブの上に半透明の凡例を表示します。
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class LegendOverlay(QWidget):
    """画像プレビュー上に重ねて表示する凡例ウィジェット。

    マウスイベントを透過し、下のウィジェットの操作を妨げません。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._layer_entries: list[tuple[tuple[int, int, int], str]] = []
        self._depth_entries: list[tuple[tuple[int, int, int], str]] = []
        self._mode: str = "none"  # "none", "layers", "depth"

    def set_layer_entries(
        self, entries: list[tuple[tuple[int, int, int], str]]
    ) -> None:
        """レイヤー凡例エントリを設定。

        Args:
            entries: (RGB色, ラベル文字列) のリスト。
        """
        self._layer_entries = entries
        self._mode = "layers"
        self.setVisible(True)
        self.update()

    def set_depth_entries(
        self, entries: list[tuple[tuple[int, int, int], str]]
    ) -> None:
        """深度凡例エントリを設定。

        Args:
            entries: (RGB色, ラベル文字列) のリスト。
        """
        self._depth_entries = entries
        self._mode = "depth"
        self.setVisible(True)
        self.update()

    def clear(self) -> None:
        """凡例を非表示にする。"""
        self._mode = "none"
        self.setVisible(False)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._mode == "none":
            return

        entries = (
            self._layer_entries if self._mode == "layers" else self._depth_entries
        )
        if not entries:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("sans-serif", 9)
        painter.setFont(font)
        fm = painter.fontMetrics()

        swatch_size = 14
        padding = 8
        row_height = max(swatch_size, fm.height()) + 4
        text_widths = [fm.horizontalAdvance(label) for _, label in entries]
        max_text_w = max(text_widths) if text_widths else 0
        box_w = padding * 2 + swatch_size + 6 + max_text_w
        box_h = padding * 2 + row_height * len(entries) - 4

        # Position at top-right
        margin = 10
        x0 = self.width() - box_w - margin
        y0 = margin

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(QRect(x0, y0, box_w, box_h), 6, 6)

        # Entries
        for i, (color, label) in enumerate(entries):
            row_y = y0 + padding + i * row_height

            if self._mode == "depth" and color == (-1, -1, -1):
                # Gradient swatch for depth
                grad_w = swatch_size * 3
                grad = QLinearGradient(
                    x0 + padding, row_y, x0 + padding + grad_w, row_y
                )
                grad.setColorAt(0.0, QColor(255, 255, 255))
                grad.setColorAt(1.0, QColor(0, 0, 0))
                painter.setBrush(grad)
                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.drawRect(
                    QRect(x0 + padding, row_y, grad_w, swatch_size)
                )
                painter.setPen(QColor(220, 220, 220))
                painter.drawText(
                    x0 + padding + grad_w + 6,
                    row_y + fm.ascent(),
                    label,
                )
            else:
                # Color swatch
                r, g, b = color
                painter.setBrush(QColor(r, g, b))
                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.drawRect(
                    QRect(x0 + padding, row_y, swatch_size, swatch_size)
                )
                painter.setPen(QColor(220, 220, 220))
                painter.drawText(
                    x0 + padding + swatch_size + 6,
                    row_y + fm.ascent(),
                    label,
                )

        painter.end()
