"""バックグラウンド処理スレッド。

ProcessingThread は GuiSettings を受け取り、パイプラインを構築・実行します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from PIL import Image

    from shadowbox.config.template import BoundingBox
    from shadowbox.gui.settings_bridge import GuiSettings


class ProcessingThread(QThread):
    """バックグラウンドで画像処理を行うスレッド。"""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        image: Image.Image,
        settings: GuiSettings,
        bbox: BoundingBox | None = None,
    ) -> None:
        super().__init__()
        self._image = image
        self._settings = settings
        self._bbox = bbox

    def run(self) -> None:
        """処理を実行。"""
        try:
            from shadowbox.factory import create_pipeline
            from shadowbox.gui.settings_bridge import (
                gui_to_process_kwargs,
                gui_to_shadowbox_settings,
            )

            self.progress.emit("パイプラインを作成中...")
            ss = gui_to_shadowbox_settings(self._settings)
            pipeline = create_pipeline(
                settings=ss,
                use_mock_depth=self._settings.use_mock_depth,
            )

            self.progress.emit("処理中...")
            kwargs = gui_to_process_kwargs(self._settings)
            if self._bbox is not None:
                kwargs["custom_bbox"] = self._bbox

            result = pipeline.process(self._image, **kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
