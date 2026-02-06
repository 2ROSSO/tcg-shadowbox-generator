"""i18n モジュールのテスト。"""

from shadowbox.gui.i18n import (
    TRANSLATIONS,
    get_language,
    set_language,
    tr,
)


class TestTr:
    def setup_method(self):
        set_language("en")

    def teardown_method(self):
        set_language("en")

    def test_english_default(self):
        assert tr("menu.file") == "File"

    def test_japanese(self):
        set_language("ja")
        assert tr("menu.file") == "ファイル"

    def test_fallback_to_english(self):
        set_language("fr")
        assert tr("menu.file") == "File"

    def test_missing_key_returns_key(self):
        assert tr("nonexistent.key") == "nonexistent.key"

    def test_format_params(self):
        result = tr("status.loaded", name="test.png")
        assert "test.png" in result

    def test_format_params_japanese(self):
        set_language("ja")
        result = tr("status.loaded", name="card.jpg")
        assert "card.jpg" in result

    def test_get_set_language(self):
        set_language("ja")
        assert get_language() == "ja"
        set_language("en")
        assert get_language() == "en"

    def test_all_keys_have_en_and_ja(self):
        for key, entry in TRANSLATIONS.items():
            assert "en" in entry, f"Missing 'en' for key '{key}'"
            assert "ja" in entry, f"Missing 'ja' for key '{key}'"


class TestLanguagePersistence:
    def setup_method(self):
        set_language("en")

    def teardown_method(self):
        set_language("en")

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        import shadowbox.gui.i18n as i18n_mod

        prefs_file = tmp_path / "language.json"
        monkeypatch.setattr(i18n_mod, "_prefs_path", lambda: prefs_file)

        set_language("ja")
        i18n_mod.save_language_preference()
        assert prefs_file.exists()

        set_language("en")
        assert get_language() == "en"

        i18n_mod.load_language_preference()
        assert get_language() == "ja"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        import shadowbox.gui.i18n as i18n_mod

        prefs_file = tmp_path / "nonexistent" / "language.json"
        monkeypatch.setattr(i18n_mod, "_prefs_path", lambda: prefs_file)

        set_language("en")
        i18n_mod.load_language_preference()
        assert get_language() == "en"

    def test_load_corrupted_file(self, tmp_path, monkeypatch):
        import shadowbox.gui.i18n as i18n_mod

        prefs_file = tmp_path / "language.json"
        prefs_file.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(i18n_mod, "_prefs_path", lambda: prefs_file)

        set_language("en")
        i18n_mod.load_language_preference()
        assert get_language() == "en"
