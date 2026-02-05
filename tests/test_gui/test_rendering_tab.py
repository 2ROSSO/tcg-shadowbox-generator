"""3D表示タブテスト。"""

import pytest

pytestmark = pytest.mark.gui


class TestRenderingTab:
    def test_defaults(self, qtbot):
        from shadowbox.gui.widgets.rendering_tab import RenderingTab

        tab = RenderingTab()
        qtbot.addWidget(tab)
        assert tab.render_mode.currentText() == "points"
        assert tab.point_size_slider.value() == 3
        assert not tab.show_axes.isChecked()
        assert tab.show_frame.isChecked()

    def test_mesh_mode_toggles_visibility(self, qtbot):
        from shadowbox.gui.widgets.rendering_tab import RenderingTab

        tab = RenderingTab()
        qtbot.addWidget(tab)
        tab.show()
        tab.render_mode.setCurrentText("mesh")
        # mesh_size should not be hidden, point_size should be hidden
        assert not tab.mesh_size.isHidden()
        assert tab.point_size_slider.isHidden()

    def test_points_mode_toggles_visibility(self, qtbot):
        from shadowbox.gui.widgets.rendering_tab import RenderingTab

        tab = RenderingTab()
        qtbot.addWidget(tab)
        tab.show()
        tab.render_mode.setCurrentText("mesh")
        tab.render_mode.setCurrentText("points")
        assert not tab.point_size_slider.isHidden()
        assert tab.mesh_size.isHidden()

    def test_get_values(self, qtbot):
        from shadowbox.gui.widgets.rendering_tab import RenderingTab

        tab = RenderingTab()
        qtbot.addWidget(tab)
        vals = tab.get_values()
        assert vals["render_mode"] == "points"
        assert vals["point_size"] == 3.0
        assert vals["background_color"] == (30, 30, 30)
        assert vals["layer_opacity"] == 1.0

    def test_opacity_slider_label_sync(self, qtbot):
        from shadowbox.gui.widgets.rendering_tab import RenderingTab

        tab = RenderingTab()
        qtbot.addWidget(tab)
        tab.opacity_slider.setValue(50)
        assert tab.opacity_label.text() == "50%"
