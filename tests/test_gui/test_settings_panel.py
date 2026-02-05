"""設定パネルテスト。"""

import pytest

from shadowbox.gui.settings_bridge import GuiSettings

pytestmark = pytest.mark.gui


class TestSettingsPanel:
    def test_has_four_tabs(self, qtbot):
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        assert panel.tabs.count() == 4

    def test_tab_names(self, qtbot):
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        names = [panel.tabs.tabText(i) for i in range(4)]
        assert names == ["処理", "レイヤー", "フレーム", "3D表示"]

    def test_get_gui_settings_returns_dataclass(self, qtbot):
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        gs = panel.get_gui_settings()
        assert isinstance(gs, GuiSettings)

    def test_default_settings_match(self, qtbot):
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        gs = panel.get_gui_settings()
        default = GuiSettings()
        assert gs.model_mode == default.model_mode
        assert gs.cumulative_layers == default.cumulative_layers
        assert gs.include_frame == default.include_frame
        assert gs.render_mode == default.render_mode

    def test_settings_changed_signal(self, qtbot):
        from shadowbox.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.settings_changed, timeout=1000):
            panel.processing_tab.mock_depth.setChecked(True)
