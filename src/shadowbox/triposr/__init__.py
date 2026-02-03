"""TripoSR 3Dメッシュ生成モジュール。

Stability AIのTripoSRを使用して、単一画像から
3Dメッシュを直接生成する機能を提供します。

Note:
    このモジュールを使用するには、triposr依存関係が必要です:
    pip install shadowbox[triposr]
"""

from shadowbox.triposr.settings import TripoSRSettings

__all__ = ["TripoSRSettings"]

# 遅延インポート（オプショナル依存関係のため）
def create_triposr_generator(settings: TripoSRSettings):
    """TripoSR生成器を作成。

    Args:
        settings: TripoSR設定。

    Returns:
        TripoSRGenerator: 設定済みの生成器。

    Raises:
        ImportError: TripoSR依存関係がインストールされていない場合。
    """
    from shadowbox.triposr.generator import TripoSRGenerator
    return TripoSRGenerator(settings)
