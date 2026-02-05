"""ダークテーマのスタイルシート定義。"""

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QGroupBox {
    color: #ddd;
    border: 1px solid #444;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
}
QLabel {
    color: #ddd;
}
QPushButton {
    background-color: #3a3a3a;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 5px 15px;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed {
    background-color: #2a2a2a;
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666;
}
QPushButton#primary {
    background-color: #2d5aa0;
    border-color: #3a6db5;
}
QPushButton#primary:hover {
    background-color: #3a6db5;
}
QPushButton#primary:disabled {
    background-color: #1a3460;
    color: #666;
}
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3a3a3a;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 2px 5px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #555;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #888;
    border: 1px solid #666;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #aaa;
}
QCheckBox {
    color: #ddd;
    spacing: 5px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #555;
    border-radius: 2px;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #2d5aa0;
    border-color: #3a6db5;
}
QTabWidget::pane {
    border: 1px solid #444;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #2a2a2a;
    color: #999;
    border: 1px solid #444;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ddd;
    border-bottom-color: #1e1e1e;
}
QTabBar::tab:hover:!selected {
    background-color: #333;
}
QMenuBar {
    background-color: #2a2a2a;
    color: #ddd;
}
QMenuBar::item:selected {
    background-color: #3a3a3a;
}
QMenu {
    background-color: #2a2a2a;
    color: #ddd;
}
QMenu::item:selected {
    background-color: #3a3a3a;
}
QStatusBar {
    background-color: #2a2a2a;
    color: #888;
}
QProgressBar {
    background-color: #2a2a2a;
    border: 1px solid #444;
    border-radius: 3px;
    text-align: center;
    color: #ddd;
}
QProgressBar::chunk {
    background-color: #2d5aa0;
    border-radius: 2px;
}
QScrollArea {
    border: none;
}
"""
