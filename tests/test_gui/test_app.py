"""MainWindow テスト。"""

import pytest

pytestmark = pytest.mark.gui


class TestShadowboxAppInit:
    def test_window_title(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.windowTitle() == "TCG Shadowbox Generator"

    def test_minimum_size(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.minimumWidth() == 1000
        assert window.minimumHeight() == 700

    def test_has_settings_panel(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.settings_panel is not None

    def test_has_action_buttons(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.action_buttons is not None

    def test_has_image_preview(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert window.image_preview is not None

    def test_generate_button_disabled_initially(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert not window.action_buttons.generate_btn.isEnabled()

    def test_view_3d_button_disabled_initially(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert not window.action_buttons.view_3d_btn.isEnabled()

    def test_export_button_disabled_initially(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert not window.action_buttons.export_btn.isEnabled()

    def test_status_bar_initial_message(self, qtbot):
        from shadowbox.gui.app import ShadowboxApp

        window = ShadowboxApp()
        qtbot.addWidget(window)
        assert "画像を読み込んでください" in window.statusBar().currentMessage()
