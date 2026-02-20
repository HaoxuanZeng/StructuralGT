# apps/view/widgets/ribbon_bar.py
"""Ribbon bar widget for StructuralGT GUI."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QWidget,
)
from service.main_controller import MainController
from view.resources import get_icon_path


class RibbonBar(QWidget):
    """Ribbon bar widget for StructuralGT GUI."""

    toggle_panel_signal = Signal()
    change_view_signal = Signal(str)
    refresh_signal = Signal()

    def __init__(self, controller: MainController, main_window):
        """Initialize the ribbon bar widget."""
        super().__init__(main_window)
        self.main_window = main_window
        self.controller = controller
        self.hide_panel = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(40)

        # Set fixed height and style
        self.setFixedHeight(40)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 0, 5, 0)
        main_layout.setSpacing(5)
        left_layout = QHBoxLayout()
        left_layout.setSpacing(5)
        left_layout.setAlignment(Qt.AlignLeft)
        right_layout = QHBoxLayout()
        right_layout.setSpacing(5)
        right_layout.setAlignment(Qt.AlignRight)

        self.toggle_panel_button = QToolButton()
        self.toggle_panel_button.setIcon(QIcon(get_icon_path("panel_close.png")))
        self.toggle_panel_button.setToolTip("Show/Hide Left Pane")
        self.toggle_panel_button.clicked.connect(self._toggle_left_panel)

        self.combo_box = QComboBox()
        self.combo_box.setFixedWidth(160)
        self.combo_box.addItems(["Raw Image", "Binarized Image", "Extracted Graph"])
        self.combo_box.currentTextChanged.connect(self.change_view_signal)
        self.combo_box.setDisabled(True)

        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(QIcon(get_icon_path("refresh.png")))
        self.refresh_button.setToolTip("Refresh")
        self.refresh_button.setDisabled(True)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)

        main_layout.addLayout(left_layout)
        left_layout.addWidget(self.toggle_panel_button)
        main_layout.addLayout(right_layout)
        right_layout.addWidget(self.combo_box)
        right_layout.addWidget(self.refresh_button)

    def _toggle_left_panel(self):
        self.toggle_panel_signal.emit()
        self.hide_panel = not self.hide_panel
        if self.hide_panel:
            self.toggle_panel_button.setIcon(QIcon(get_icon_path("panel_open.png")))
        else:
            self.toggle_panel_button.setIcon(QIcon(get_icon_path("panel_close.png")))

    def _on_refresh_clicked(self):
        self.refresh_signal.emit()
