"""ProcessingThread テスト。"""

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.gui


class TestProcessingThread:
    def test_creates_thread(self, qtbot):
        from shadowbox.gui.processing import ProcessingThread
        from shadowbox.gui.settings_bridge import GuiSettings

        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        gs = GuiSettings(use_mock_depth=True)
        thread = ProcessingThread(image=img, settings=gs)
        assert thread is not None

    def test_mock_processing_succeeds(self, qtbot):
        from shadowbox.gui.processing import ProcessingThread
        from shadowbox.gui.settings_bridge import GuiSettings

        img = Image.fromarray(
            np.random.randint(0, 255, (80, 60, 3), dtype=np.uint8)
        )
        gs = GuiSettings(use_mock_depth=True, num_layers=3)
        thread = ProcessingThread(image=img, settings=gs)

        results = []
        thread.finished.connect(lambda r: results.append(r))

        with qtbot.waitSignal(thread.finished, timeout=30000):
            thread.start()

        assert len(results) == 1
        assert results[0].mesh is not None
