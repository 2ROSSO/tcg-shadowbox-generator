"""YAML設定ローダーモジュール。

このモジュールは、カードテンプレートをYAMLファイルとして
読み書きするためのローダークラスを提供します。
"""

from pathlib import Path
from typing import Protocol

import yaml

from shadowbox.config.template import CardTemplate


class ConfigLoaderProtocol(Protocol):
    """設定ローダーのプロトコル (DI用インターフェース)。

    依存性注入パターンに基づき、設定ローダーの
    インターフェースを定義します。テスト時のモック化や
    異なる実装への差し替えを容易にします。
    """

    def load_template(self, name: str) -> CardTemplate:
        """名前でカードテンプレートを読み込む。

        Args:
            name: テンプレート名。

        Returns:
            読み込んだCardTemplate。
        """
        ...

    def save_template(self, template: CardTemplate) -> None:
        """カードテンプレートをファイルに保存。

        Args:
            template: 保存するテンプレート。
        """
        ...

    def list_templates(self) -> list[str]:
        """利用可能なテンプレート名の一覧を取得。

        Returns:
            テンプレート名のリスト。
        """
        ...


class YAMLConfigLoader:
    """YAML形式の設定ローダー実装。

    カードテンプレートをYAMLファイルとして保存・読み込みします。
    テンプレートは指定されたディレクトリに「テンプレート名.yaml」
    という形式で保存されます。

    Example:
        >>> loader = YAMLConfigLoader(Path("data/templates"))
        >>>
        >>> # テンプレート一覧を取得
        >>> templates = loader.list_templates()
        >>> print(templates)  # ['pokemon_standard', 'mtg_standard']
        >>>
        >>> # テンプレートを読み込み
        >>> template = loader.load_template("pokemon_standard")
    """

    def __init__(self, templates_dir: Path) -> None:
        """ローダーを初期化。

        Args:
            templates_dir: テンプレートYAMLファイルを格納するディレクトリ。
        """
        self._templates_dir = Path(templates_dir)

    @property
    def templates_dir(self) -> Path:
        """テンプレートディレクトリのパスを返す。"""
        return self._templates_dir

    def load_template(self, name: str) -> CardTemplate:
        """名前でカードテンプレートを読み込む。

        Args:
            name: テンプレート名 (.yaml拡張子は不要)。

        Returns:
            読み込んだCardTemplateインスタンス。

        Raises:
            FileNotFoundError: テンプレートファイルが存在しない場合。
            ValueError: テンプレートファイルが空または不正な場合。
        """
        file_path = self._templates_dir / f"{name}.yaml"

        if not file_path.exists():
            raise FileNotFoundError(f"テンプレートが見つかりません: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"テンプレートファイルが空です: {file_path}")

        return CardTemplate.from_dict(data)

    def save_template(self, template: CardTemplate) -> None:
        """カードテンプレートをYAMLファイルに保存。

        テンプレートディレクトリが存在しない場合は自動的に作成されます。

        Args:
            template: 保存するCardTemplate。
        """
        # ディレクトリが存在しない場合は作成
        self._templates_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._templates_dir / f"{template.name}.yaml"
        data = template.to_dict()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def list_templates(self) -> list[str]:
        """利用可能なテンプレート名の一覧を取得。

        Returns:
            テンプレート名のリスト (.yaml拡張子は除く)。
            ディレクトリが存在しない場合は空リスト。
        """
        if not self._templates_dir.exists():
            return []

        return [f.stem for f in self._templates_dir.glob("*.yaml")]

    def template_exists(self, name: str) -> bool:
        """テンプレートが存在するか確認。

        Args:
            name: 確認するテンプレート名。

        Returns:
            存在する場合True、しない場合False。
        """
        file_path = self._templates_dir / f"{name}.yaml"
        return file_path.exists()

    def delete_template(self, name: str) -> bool:
        """テンプレートを削除。

        Args:
            name: 削除するテンプレート名。

        Returns:
            削除成功時True、テンプレートが存在しなかった場合False。
        """
        file_path = self._templates_dir / f"{name}.yaml"

        if file_path.exists():
            file_path.unlink()
            return True

        return False
