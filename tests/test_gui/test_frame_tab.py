"""フレームタブテスト。"""

import pytest

pytestmark = pytest.mark.gui


class TestFrameTab:
    def test_defaults(self, qtbot):
        from shadowbox.gui.widgets.frame_tab import FrameTab

        tab = FrameTab()
        qtbot.addWidget(tab)
        assert tab.include_frame.isChecked()
        assert not tab.include_card_frame.isChecked()
        assert tab.frame_wall_mode.currentText() == "outer"

    def test_disable_frame_disables_children(self, qtbot):
        from shadowbox.gui.widgets.frame_tab import FrameTab

        tab = FrameTab()
        qtbot.addWidget(tab)
        tab.include_frame.setChecked(False)
        assert not tab.frame_depth.isEnabled()
        assert not tab.frame_wall_mode.isEnabled()

    def test_get_values(self, qtbot):
        from shadowbox.gui.widgets.frame_tab import FrameTab

        tab = FrameTab()
        qtbot.addWidget(tab)
        vals = tab.get_values()
        assert vals["include_frame"] is True
        assert vals["frame_depth"] == pytest.approx(0.5)
