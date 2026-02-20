"""Settings dialog for StructuralGT GUI."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from service.settings_service import SettingsService


class SettingsDialog(QDialog):
    """Settings dialog for StructuralGT GUI."""

    def __init__(self, settings_service: SettingsService, parent):
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.settings_service = settings_service
        self.setWindowTitle("Settings")
        self.setMinimumSize(480, 360)
        self.resize(480, 360)

        layout = QVBoxLayout(self, alignment=Qt.AlignTop)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        button_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        button_layout.setSpacing(20)

        info_label = QLabel("Settings", self)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.accept)

        button_layout.addWidget(self.close_button)
        layout.addStretch(1)
        layout.addLayout(button_layout)
