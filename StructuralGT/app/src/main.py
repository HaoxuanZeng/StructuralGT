"""PySide6 Application for StructualGT."""

import sys
from pathlib import Path

from controller.app_controller import AppController
from model.handler import HandlerRegistry
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from service.settings_service import SettingsService
from view.main_window import MainWindow
from view.resources import get_app_icon_path

app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))


def load_stylesheet() -> str:
    """Load custom stylesheet."""
    style_file = Path(__file__).parent / "view" / "resources" / "style" / "custom_styles.qss"
    if style_file.exists():
        return style_file.read_text(encoding="utf-8")
    return ""


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet(load_stylesheet())
    app.setWindowIcon(QIcon(get_app_icon_path()))

    settings_service = SettingsService()
    handler_registry = HandlerRegistry()
    controller = AppController(handler_registry)
    window = MainWindow(controller, settings_service=settings_service)
    window.show()

    def cleanup():
        """Clean up threads before exit."""
        controller.cleanup_threads()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())
