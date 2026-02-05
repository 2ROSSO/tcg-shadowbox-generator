"""処理タブテスト。"""

import pytest

pytestmark = pytest.mark.gui


class TestProcessingTab:
    def test_default_model_mode(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        assert tab.model_mode.currentText() == "depth"

    def test_default_num_layers_auto(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        assert tab.num_layers_auto.isChecked()
        assert not tab.num_layers.isEnabled()

    def test_uncheck_auto_enables_spinbox(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        tab.num_layers_auto.setChecked(False)
        assert tab.num_layers.isEnabled()

    def test_raw_depth_disables_layers(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        tab.raw_depth.setChecked(True)
        assert tab.depth_scale.isEnabled()
        assert not tab.num_layers_auto.isEnabled()

    def test_get_values_keys(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        vals = tab.get_values()
        expected_keys = {
            "model_mode", "detection_method", "use_mock_depth",
            "use_raw_depth", "depth_scale", "num_layers", "max_resolution",
        }
        assert set(vals.keys()) == expected_keys

    def test_get_values_auto_layers_none(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        vals = tab.get_values()
        assert vals["num_layers"] is None

    def test_settings_changed_signal(self, qtbot):
        from shadowbox.gui.widgets.processing_tab import ProcessingTab

        tab = ProcessingTab()
        qtbot.addWidget(tab)
        with qtbot.waitSignal(tab.settings_changed, timeout=1000):
            tab.model_mode.setCurrentText("triposr")
