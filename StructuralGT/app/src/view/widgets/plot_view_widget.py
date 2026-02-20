"""Plot view widget for StructuralGT GUI."""

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from model.handler import NetworkHandler
from model.handler_list_model import HandlerListModel
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)
from service.main_controller import MainController
from view.widgets.project_widget import HandlerListItemWidget


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for the plot view widget."""

    def __init__(self, parent, dpi=300):
        """Initialize an empty matplotlib canvas."""
        self._dpi = dpi
        w, h = parent.width(), parent.height()
        self.fig = Figure(
            figsize=(w / dpi, h / dpi),
            dpi=dpi,
        )
        super().__init__(self.fig)


class PlotViewWidget(QWidget):
    """Plot view widget for StructuralGT GUI."""

    def __init__(self, parent, controller: MainController):
        """Initialize the plot view widget."""
        super().__init__(parent)
        self.controller = controller
        self.model = HandlerListModel(controller.handler_registry)
        self.plot_opts = [
            ("Skeleton Plot", "skeleton"),
            ("Graph Plot", "graph"),
            ("Degree Heatmap", "degree_heatmap"),
            ("Betweenness Centrality Heatmap", "betweenness_centrality_heatmap"),
            ("Closeness Centrality Heatmap", "closeness_centrality_heatmap"),
            ("Degree Distribution", "degree_distribution"),
            (
                "Betweenness Centrality Distribution",
                "betweenness_centrality_distribution",
            ),
            ("Closeness Centrality Distribution", "closeness_centrality_distribution"),
        ]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.list_widget = QListWidget(self)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setDisabled(True)

        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setDisabled(True)
        self.combo_box = QComboBox(self)
        self.combo_box.addItems([opt[0] for opt in self.plot_opts])
        self.combo_box.currentIndexChanged.connect(self._on_combo_changed)
        self.combo_box.setDisabled(True)

        left_layout.addWidget(self.combo_box)
        left_layout.addWidget(self.list_widget)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout, 1)
        self.setLayout(layout)

        self._populate_list()

    def set_plot(self, plot_name: str) -> None:
        """Set the plot."""
        if not plot_name:
            self.clear()
            return
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self.controller.plot_selected_handler(plot_name, ax)
        self.canvas.draw()

    def clear(self) -> None:
        """Clear the plot."""
        self.combo_box.setCurrentIndex(0)
        self.canvas.fig.clear()
        self.canvas.draw()

    def refresh(self) -> None:
        """Refresh the plot."""
        self.model.refresh()
        self._populate_list()
        self.canvas.fig.clear()
        self.canvas.draw()
        if self.model.rowCount() == 0:
            self.combo_box.setDisabled(True)
            self.list_widget.setDisabled(True)
            self.toolbar.setDisabled(True)
        else:
            self.combo_box.setEnabled(True)
            self.list_widget.setEnabled(True)
            self.toolbar.setEnabled(True)

    def _populate_list(self):
        self.list_widget.clear()
        row_count = self.model.rowCount()
        for i in range(row_count):
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 80))
            widget = HandlerListItemWidget(self.model, i, self.list_widget)
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def _on_item_clicked(self, item: QListWidgetItem):
        row = self.list_widget.row(item)
        handler_index = self.model.data(self.model.index(row, 0), Qt.UserRole)
        if handler_index is not None:
            self.controller.set_selected_index(handler_index)
        self._on_combo_changed(self.combo_box.currentIndex())

    def _on_combo_changed(self, index: int) -> None:
        """Handle combo box selection change."""
        handler = self.controller.get_selected_handler()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        _, plot_name = self.plot_opts[index]
        self.set_plot(plot_name)
