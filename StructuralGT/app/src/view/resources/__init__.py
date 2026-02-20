"""Resources module for StructuralGT GUI."""

from .resources import *  # noqa: F403
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette

RESOURCE_PREFIX = ":/resources"


def get_resource_path(*path_parts: str) -> str:
    """Get resource path using QRC format."""
    path = "/".join(path_parts)
    return f"{RESOURCE_PREFIX}/{path}"


def get_app_icon_path() -> str:
    """Get app icon resource path."""
    return get_resource_path("icons", "StructuralGT.png")


def is_dark_theme() -> bool:
    """Detect if the system is using dark theme."""
    app = QApplication.instance()
    if app:
        palette = app.palette()
        window_color = palette.color(QPalette.Window)
        return window_color.lightness() < 128
    return False


def get_icon_path(icon_name: str) -> str:
    """Get icon resource path based on system theme."""
    theme = "dark" if is_dark_theme() else "light"
    return get_resource_path("icons", theme, icon_name)
