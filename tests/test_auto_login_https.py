import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

for env_var in [
    "REDDIT",
    "SHOPPING",
    "SHOPPING_ADMIN",
    "GITLAB",
    "WIKIPEDIA",
    "MAP",
    "HOMEPAGE",
]:
    os.environ.setdefault(env_var, f"http://{env_var.lower()}.example")

from browser_env import auto_login  # noqa: E402


def _mock_playwright() -> tuple[Mock, Mock, Mock]:
    page = Mock()
    page.url = "http://shopping.example/wishlist/"
    page.content.return_value = ""

    context = Mock()
    context.new_page.return_value = page

    browser = Mock()
    browser.new_context.return_value = context

    playwright = Mock()
    playwright.chromium.launch.return_value = browser

    manager = MagicMock()
    manager.__enter__.return_value = playwright
    return manager, playwright, browser


def test_is_expired_ignores_https_errors_for_context(tmp_path: Path) -> None:
    manager, playwright, browser = _mock_playwright()
    storage_state = tmp_path / "existing-auth-state.json"
    storage_state.write_text("{}")

    with patch.object(auto_login, "sync_playwright", return_value=manager):
        auto_login.is_expired(
            storage_state,
            "http://shopping.example/wishlist/",
            "",
        )

    playwright.chromium.launch.assert_called_once_with(
        headless=True,
        slow_mo=auto_login.SLOW_MO,
        args=auto_login.CHROMIUM_ARGS,
    )
    browser.new_context.assert_called_once_with(
        storage_state=storage_state,
        ignore_https_errors=True,
    )


def test_renew_comb_ignores_https_errors_for_context() -> None:
    manager, playwright, browser = _mock_playwright()

    with patch.object(auto_login, "sync_playwright", return_value=manager):
        auto_login.renew_comb(["shopping"], auth_folder="/tmp")

    playwright.chromium.launch.assert_called_once_with(
        headless=auto_login.HEADLESS,
        args=auto_login.CHROMIUM_ARGS,
    )
    browser.new_context.assert_called_once_with(ignore_https_errors=True)
