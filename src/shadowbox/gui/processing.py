"""バックグラウンド処理スレッド。

ProcessingThread は GuiSettings を受け取り、パイプラインを構築・実行します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

from shadowbox.gui.i18n import tr

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

            self.progress.emit(tr("progress.creating"))
            ss = gui_to_shadowbox_settings(self._settings)
            pipeline = create_pipeline(
                settings=ss,
                use_mock_depth=self._settings.use_mock_depth,
            )

            self.progress.emit(tr("progress.processing"))
            kwargs = gui_to_process_kwargs(self._settings)

            # auto_detect は ProcessingThread 側で処理するため除去
            kwargs.pop("auto_detect", None)

            if self._bbox is not None:
                # ユーザーが手動選択済み → そのまま使用
                bbox_key = (
                    "bbox"
                    if self._settings.model_mode == "triposr"
                    else "custom_bbox"
                )
                kwargs[bbox_key] = self._bbox
            elif self._settings.detection_method != "none":
                # RegionDetector で検出
                self.progress.emit(tr("progress.detecting"))
                from shadowbox.detection.region import (
                    DETECTION_METHODS,
                    RegionDetector,
                )

                detector = RegionDetector()
                method = self._settings.detection_method
                det_method = (
                    method if method in DETECTION_METHODS else None
                )
                result = detector.detect(self._image, method=det_method)
                if result.confidence > 0:
                    bbox_key = (
                        "bbox"
                        if self._settings.model_mode == "triposr"
                        else "custom_bbox"
                    )
                    kwargs[bbox_key] = result.bbox

            result = pipeline.process(self._image, **kwargs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
