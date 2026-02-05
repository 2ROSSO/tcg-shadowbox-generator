"""レイヤータブテスト。"""

import pytest

pytestmark = pytest.mark.gui


class TestLayersTab:
    def test_defaults(self, qtbot):
        from shadowbox.gui.widgets.layers_tab import LayersTab

        tab = LayersTab()
        qtbot.addWidget(tab)
        assert tab.cumulative_layers.isChecked()
        assert tab.back_panel.isChecked()
        assert tab.layer_interpolation.value() == 0
        assert tab.spacing_mode.currentText() == "even"

    def test_pop_out_slider_sync(self, qtbot):
        from shadowbox.gui.widgets.layers_tab import LayersTab

        tab = LayersTab()
        qtbot.addWidget(tab)
        tab.pop_out_slider.setValue(50)
        assert abs(tab.pop_out_spin.value() - 0.5) < 0.02

    def test_get_values(self, qtbot):
        from shadowbox.gui.widgets.layers_tab import LayersTab

        tab = LayersTab()
        qtbot.addWidget(tab)
        vals = tab.get_values()
        assert vals["cumulative_layers"] is True
        assert vals["layer_thickness"] == pytest.approx(0.1)

    def test_settings_changed_signal(self, qtbot):
        from shadowbox.gui.widgets.layers_tab import LayersTab

        tab = LayersTab()
        qtbot.addWidget(tab)
        with qtbot.waitSignal(tab.settings_changed, timeout=1000):
            tab.cumulative_layers.setChecked(False)
