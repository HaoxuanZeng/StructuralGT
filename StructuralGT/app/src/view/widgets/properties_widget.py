"""Properties widget for StructuralGT GUI."""

import pathlib
from ctypes import alignment

import pandas as pd
from model.handler import NetworkHandler, PointNetworkHandler
from model.table_model import TableModel
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from service.main_controller import MainController
from view.dialogs.file_dialog import export_file


class PropertiesWidget(QWidget):
    """Properties widget for StructuralGT GUI."""

    def __init__(self, controller: MainController, parent):
        """Initialize the properties widget."""
        super().__init__(parent)
        self.controller = controller
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignTop)

        self.empty_label = QLabel("No properties to show.", self)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setVisible(False)
        self.empty_label.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.empty_label)

        self.image_label = QLabel("Image Properties", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.image_label)

        self.image_table = QTableView(self)
        self.image_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_table.setContentsMargins(5, 5, 5, 5)
        self.image_table.horizontalHeader().setVisible(False)
        self.image_table.verticalHeader().setVisible(False)
        self.image_table.setShowGrid(True)
        self.image_table.setAlternatingRowColors(True)
        self.image_table.setSelectionBehavior(QTableView.SelectRows)
        self.image_table.setEditTriggers(QTableView.NoEditTriggers)
        self.image_table.horizontalHeader().setStretchLastSection(True)
        self.image_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_model = TableModel()
        self.image_table.setModel(self.image_model)
        layout.addWidget(self.image_table)

        layout.addSpacing(10)

        self.graph_label = QLabel("Graph Properties", self)
        self.graph_label.setContentsMargins(5, 5, 5, 5)
        self.graph_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.graph_label)

        self.graph_table = QTableView(self)
        self.graph_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.graph_table.setContentsMargins(5, 5, 5, 5)
        self.graph_table.horizontalHeader().setVisible(False)
        self.graph_table.verticalHeader().setVisible(False)
        self.graph_table.setShowGrid(True)
        self.graph_table.setAlternatingRowColors(True)
        self.graph_table.setSelectionBehavior(QTableView.SelectRows)
        self.graph_table.setEditTriggers(QTableView.NoEditTriggers)
        self.graph_table.horizontalHeader().setStretchLastSection(True)
        self.graph_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graph_model = TableModel()
        self.graph_table.setModel(self.graph_model)
        layout.addWidget(self.graph_table)

        self.export_button = QPushButton("Export All", self)
        self.export_button.clicked.connect(self._on_export_button_clicked)
        layout.addWidget(self.export_button)

        self.controller.handler_changed_signal.connect(self.refresh)
        self.controller.compute_graph_properties_finished_signal.connect(
            self._on_compute_finished
        )
        self.controller.extract_graph_finished_signal.connect(self.refresh)

        self.refresh()

    def _on_export_button_clicked(self):
        handler = self.controller.get_selected_handler()
        if handler:
            file_path = export_file(
                parent=self,
                default_filename="properties.xlsx",
                file_filter="Excel (*.xlsx)",
                default_dir=str(handler["paths"]["input_dir"]),
            )
            if file_path:
                image_properties = self._get_image_properties(handler)
                graph_properties = self._get_graph_properties(handler)
                properties = image_properties + graph_properties
                with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                    df = pd.DataFrame(properties, columns=["Property", "Value"])
                    df.to_excel(writer, index=False, sheet_name="Properties")

    def _on_compute_finished(self, success: bool):
        self.refresh()
        if success and self.controller:
            parent = self.parent().parent()
            parent.setCurrentIndex(1)

    def refresh(self):
        """Refresh the properties widget."""
        if self.controller.handler_registry.count() == 0:
            self.empty_label.setVisible(True)
            self.image_label.setVisible(False)
            self.image_table.setVisible(False)
            self.graph_label.setVisible(False)
            self.graph_table.setVisible(False)
            self.export_button.setVisible(False)
            return

        self.empty_label.setVisible(False)
        handler = self.controller.get_selected_handler()

        # Update Image Properties
        image_data = self._get_image_properties(handler)
        self.image_model.setData(image_data)
        self.image_table.resizeColumnsToContents()
        if image_data:
            self.image_label.setVisible(True)
            self.image_table.setVisible(True)
            self.export_button.setVisible(True)
            row_height = self.image_table.verticalHeader().defaultSectionSize()
            self.image_table.setFixedHeight(5 + len(image_data) * row_height)
        else:
            self.image_label.setVisible(False)
            self.image_table.setVisible(False)
            self.export_button.setVisible(False)
            self.image_table.setFixedHeight(0)

        # Update Graph Properties
        graph_data = self._get_graph_properties(handler)
        self.graph_model.setData(graph_data)
        if graph_data:
            self.graph_label.setVisible(True)
            self.graph_table.setVisible(True)
            self.graph_table.resizeColumnsToContents()
            row_height = self.graph_table.verticalHeader().defaultSectionSize()
            self.graph_table.setFixedHeight(5 + len(graph_data) * row_height)
        else:
            self.graph_label.setVisible(False)
            self.graph_table.setVisible(False)

    def _get_image_properties(self, handler):
        if not handler:
            return []

        data = []

        # Name
        input_dir = handler["paths"]["input_dir"]
        name = pathlib.Path(input_dir).name
        data.append(["Name", name])

        # Size or Point Count
        if isinstance(handler, NetworkHandler):
            image_shape = handler["ui_properties"].get("image_shape")
            if image_shape:
                dim = handler["network_properties"].get("dim", None)
                if dim == 2:
                    # 2D: width x height
                    data.append(["Size", f"{image_shape[1]} × {image_shape[0]}"])
                else:
                    # 3D: width x height x depth
                    data.append(
                        [
                            "Size",
                            f"{image_shape[2]} × {image_shape[1]} ×{image_shape[0]}",
                        ]
                    )
            else:
                data.append(["Size", "N/A"])
        elif isinstance(handler, PointNetworkHandler):
            # Point Network: cutoff
            data.append(["Cutoff", f"{handler['ui_properties']['cutoff']}"])

        # Dimensions
        dim = handler["network_properties"].get("dim")
        if dim is not None:
            data.append(["Dimensions", f"{dim}D"])
        else:
            data.append(["Dimensions", "N/A"])

        return data

    def _get_graph_properties(self, handler):
        """Get graph properties data for the table."""
        if not handler:
            return []

        if not handler["ui_properties"].get("extracted_loaded", False):
            return []

        data = []

        # Edge Count and Node Count
        if handler["network"] and hasattr(handler["network"], "graph"):
            graph = handler["network"].graph
            data.append(["Number of Edges", f"{graph.ecount()}"])
            data.append(["Number of Nodes", f"{graph.vcount()}"])
            data.append(
                ["Weight Type", str(handler["network_properties"]["weight_type"])]
            )

        # Network properties
        network_properties = handler["network_properties"]
        property_names = {
            "diameter": "Network Diameter",
            "density": "Graph Density",
            "average_clustering_coefficient": "Average Clustering Coefficient",
            "assortativity": "Assortativity Coefficient",
            "average_closeness": "Average Closeness Centrality",
            "average_degree": "Average Degree",
            "nematic_order_parameter": "Nematic Order Parameter",
            "average_betweenness_centrality": "Average Betweenness Centrality",
            "effective_resistance": "Effective Resistance",
        }

        for key, display_name in property_names.items():
            value = network_properties.get(key)
            if value is not None:
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.5f}"
                    data.append([display_name, formatted_value])
                else:
                    data.append([display_name, str(value)])

        return data
