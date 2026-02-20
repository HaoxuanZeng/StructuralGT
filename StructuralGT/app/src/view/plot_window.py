"""Plot window for StructuralGT GUI."""

from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from service.main_controller import MainController

from view.widgets.plot_view_widget import PlotViewWidget


class PlotWindow(QMainWindow):
    """Plot window for StructuralGT GUI."""

    def __init__(self, controller: MainController, parent):
        """Initialize the plot window."""
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Plotting")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        self.plot_view = PlotViewWidget(self, controller)
        layout.addWidget(self.plot_view)
