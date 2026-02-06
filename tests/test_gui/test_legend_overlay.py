"""凡例オーバーレイのテスト。"""

import pytest

pytestmark = pytest.mark.gui


class TestLegendOverlay:
    def test_initial_hidden(self, qtbot):
        from shadowbox.gui.widgets.legend_overlay import LegendOverlay

        overlay = LegendOverlay()
        qtbot.addWidget(overlay)
        assert overlay._mode == "none"

    def test_set_layer_entries(self, qtbot):
        from shadowbox.gui.widgets.legend_overlay import LegendOverlay

        overlay = LegendOverlay()
        qtbot.addWidget(overlay)
        entries = [
            ((255, 0, 0), "Layer 0 (near)"),
            ((0, 255, 0), "Layer 1"),
            ((0, 0, 255), "Layer 2 (far)"),
        ]
        overlay.set_layer_entries(entries)
        assert overlay._mode == "layers"
        assert overlay.isVisible()
        assert len(overlay._layer_entries) == 3

    def test_set_depth_entries(self, qtbot):
        from shadowbox.gui.widgets.legend_overlay import LegendOverlay

        overlay = LegendOverlay()
        qtbot.addWidget(overlay)
        entries = [
            ((255, 255, 255), "Near"),
            ((128, 128, 128), "Mid"),
            ((0, 0, 0), "Far"),
        ]
        overlay.set_depth_entries(entries)
        assert overlay._mode == "depth"
        assert overlay.isVisible()

    def test_clear(self, qtbot):
        from shadowbox.gui.widgets.legend_overlay import LegendOverlay

        overlay = LegendOverlay()
        qtbot.addWidget(overlay)
        overlay.set_layer_entries([((255, 0, 0), "Layer 0")])
        assert overlay.isVisible()
        overlay.clear()
        assert overlay._mode == "none"
        assert not overlay.isVisible()

    def test_paint_does_not_crash(self, qtbot):
        from shadowbox.gui.widgets.legend_overlay import LegendOverlay

        overlay = LegendOverlay()
        qtbot.addWidget(overlay)
        overlay.resize(400, 300)
        overlay.set_layer_entries([
            ((255, 0, 0), "Layer 0 (near)"),
            ((0, 255, 0), "Layer 1 (far)"),
        ])
        # Force a paint event
        overlay.repaint()

    def test_paint_depth_gradient(self, qtbot):
        from shadowbox.gui.widgets.legend_overlay import LegendOverlay

        overlay = LegendOverlay()
        qtbot.addWidget(overlay)
        overlay.resize(400, 300)
        overlay.set_depth_entries([
            ((-1, -1, -1), "Near → Far"),
            ((128, 128, 128), "Mid"),
        ])
        overlay.repaint()
