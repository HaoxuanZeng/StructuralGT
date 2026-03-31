"""Main controller for StructuralGT GUI."""

import logging
import uuid
from datetime import datetime

import numpy as np
from model.handler import HandlerRegistry, NetworkHandler, PointNetworkHandler
from model.task_list_model import Task
from PySide6.QtCore import QObject
from service.network_service import NetworkService
from service.ui_service import UIService

from controller.event_bus import EventBus
from controller.worker import Worker

logger = logging.getLogger(__name__)


class AppController(QObject):
    """Main controller for StructuralGT GUI."""

    def __init__(self, handler_registry: HandlerRegistry) -> None:
        """Initialize the main controller."""
        super().__init__()
        self.handler_registry = handler_registry
        self._active_threads: list[Worker] = []
        self._event_bus = EventBus.get_instance()
        logger.info("AppController initialized")

    def add_network(self, folder_path: str, dim: int) -> None:
        """Add a network to the handler registry.

        Args:
            folder_path: Path to the folder containing images.
            dim: Dimensionality (2 or 3).
        """
        try:
            NetworkService.add_network(self.handler_registry, folder_path, dim)
        except Exception as e:
            self._event_bus.emit_alert("Failed to add Network", repr(e))
            return
        self._event_bus.emit_handler_changed()
        self._event_bus.emit_view_changed("Raw Image")

    def add_point_network(self, file_path: str) -> None:
        """Add a point network to the handler registry."""
        try:
            NetworkService.add_point_network(self.handler_registry, file_path)
        except Exception as e:
            self._event_bus.emit_alert("Failed to add Point Network", repr(e))
            return
        self.handler_changed_signal.emit()
        pipeline = UIService.get_selected_extracted_graph(self.handler_registry)
        self._event_bus.emit_extract_graph_finished(pipeline)
        self._event_bus.emit_view_changed("Extracted Graph Only")

    def delete_network(self, index: int):
        """Delete a network from the handler registry."""
        try:
            NetworkService.delete_network(self.handler_registry, index)
        except Exception as e:
            self._event_bus.emit_alert("Failed to delete Network/Point Network", repr(e))
            return
        self._event_bus.emit_handler_changed()
        if self.handler_registry.count() == 0:
            self._event_bus.emit_view_changed("Welcome Page")

    def get_selected_handler(self):
        """Get the selected handler from the handler registry."""
        return self.handler_registry.get_selected()

    def set_selected_index(self, index: int):
        """Set the selected index of the handler registry."""
        try:
            UIService.set_selected_index(self.handler_registry, index)
        except Exception as e:
            self._event_bus.emit_alert("Failed to set selected index", repr(e))
            return
        self._event_bus.emit_handler_changed()

        handler = self.handler_registry.get(index)
        if handler is not None:
            if isinstance(handler, NetworkHandler):
                self._event_bus.emit_view_changed("Raw Image")
            elif isinstance(handler, PointNetworkHandler):
                pipeline = self.get_selected_extracted_graph()
                if pipeline:
                    self._event_bus.emit_extract_graph_finished(pipeline)
                self._event_bus.emit_view_changed("Extracted Graph Only")

    def set_selected_slice_index(self, index: int) -> None:
        """Set the selected slice index of the selected handler."""
        try:
            UIService.set_selected_slice_index(self.handler_registry, index)
        except Exception as e:
            self._event_bus.emit_alert("Failed to set selected slice index", repr(e))
            return
        self._event_bus.emit_handler_changed()

    def get_selected_slice_raw_image(self, index: int) -> np.ndarray | None:
        """Get the selected slice raw image from the selected handler."""
        try:
            return UIService.get_selected_slice_raw_image(
                self.handler_registry, index
            )
        except Exception as e:
            self._event_bus.emit_alert("Failed to get selected slice raw image", repr(e))
            return None

    def get_selected_slice_binarized_image(self, index: int) -> np.ndarray | None:
        """Get the selected slice binarized image from the selected handler."""
        try:
            return UIService.get_selected_slice_binarized_image(
                self.handler_registry, index
            )
        except Exception as e:
            self._event_bus.emit_alert(
                "Failed to get selected slice binarized image", repr(e)
            )
            return None

    def get_selected_extracted_graph(self) -> str | None:
        """Get the selected extracted graph from the selected handler."""
        try:
            return UIService.get_selected_extracted_graph(self.handler_registry)
        except Exception as e:
            self._event_bus.emit_alert("Failed to get selected extracted graph", repr(e))
            return None

    def plot_selected_handler(self, plot_name: str, ax):
        """Plot the selected handler."""
        dispatch = {
            "skeleton": NetworkService.plot_skeleton,
            "graph": NetworkService.plot_graph,
            "degree_heatmap": NetworkService.plot_degree_heatmap,
            "betweenness_centrality_heatmap": NetworkService.plot_betweenness_centrality_heatmap,
            "closeness_centrality_heatmap": NetworkService.plot_closeness_centrality_heatmap,
            "degree_distribution": NetworkService.plot_degree_distribution,
            "betweenness_centrality_distribution": NetworkService.plot_betweenness_centrality_distribution,
            "closeness_centrality_distribution": NetworkService.plot_closeness_centrality_distribution,
        }
        fn = dispatch.get(plot_name)
        if fn is None:
            self._emit_alert("Invalid plot", f"Unknown plot type: {plot_name}")
            return
        try:
            fn(self.handler_registry, ax)
        except Exception as e:
            self._event_bus.emit_alert("Failed to plot", repr(e))

    def binarize_selected_network(self, options: dict) -> None:
        """Binarize the selected network in a background thread.

        Args:
            options: Binarization options dictionary.
        """
        if self.handler_registry.task_count() >= 10:
            self._event_bus.emit_alert(
                "Task limit reached", "Maximum of 10 concurrent tasks allowed."
            )
            return

        handler = self.handler_registry.get_selected()
        if not handler or handler["tasks"]["Binarize"]:
            return

        task = Task(task_id=str(uuid.uuid4()), task_type="Binarize")
        handler["tasks"]["Binarize"] = task
        task.status = "Running"
        self._event_bus.emit_task_changed()
        logger.info(f"Task {task.task_id} created for binarization")

        def on_finished(result, error=None):
            """Send signals to the main thread."""
            handler["tasks"]["Binarize"] = None
            self._event_bus.emit_task_changed()

            if error:
                logger.error(f"Task {task.task_id} failed: {error}")
                self._event_bus.emit_alert("Failed to binarize network", repr(error))
                self._event_bus.emit_binarize_finished(False)
            else:
                self._event_bus.emit_handler_changed()
                self._event_bus.emit_binarize_finished(True)

        worker = Worker(
            NetworkService.binarize_selected_network,
            callback=on_finished,
            task=task,
            handler_registry=self.handler_registry,
            options=options,
        )
        task.thread = worker
        self._active_threads.append(worker)
        worker.finished.connect(lambda: self._active_threads.remove(worker))
        worker.start()

    def extract_graph_from_selected_network(self, weight_type=None):
        """Extract the graph from the selected network."""
        if self.handler_registry.task_count() >= 10:
            self._event_bus.emit_alert(
                "Task limit reached", "Maximum of 10 concurrent tasks allowed."
            )
            return

        handler = self.handler_registry.get_selected()
        if not handler or handler["tasks"]["Extract Graph"]:
            return

        task = Task(task_id=str(uuid.uuid4()), task_type="Extract Graph")
        handler["tasks"]["Extract Graph"] = task
        task.status = "Running"
        self._event_bus.emit_task_changed()
        logger.info(f"Task {task.task_id} created for graph extraction")

        def on_finished(result, error=None):
            """Send signals to the main thread."""
            handler["tasks"]["Extract Graph"] = None
            self._event_bus.emit_task_changed()
            if error:
                logger.error(f"Task {task.task_id} failed: {error}")
                self._event_bus.emit_alert("Failed to extract graph", repr(error))
                self._event_bus.emit_extract_graph_finished(None)
            else:
                pipeline = self.get_selected_extracted_graph()
                self._event_bus.emit_handler_changed()
                self._event_bus.emit_extract_graph_finished(pipeline)

        worker = Worker(
            NetworkService.extract_graph_from_selected_network,
            callback=on_finished,
            handler_registry=self.handler_registry,
            weight_type=weight_type,
            task=task,
        )
        task.thread = worker
        self._active_threads.append(worker)
        worker.finished.connect(lambda: self._active_threads.remove(worker))
        worker.start()

    def compute_graph_properties_from_selected_network(self, options: dict):
        """Compute the graph properties of the selected network."""
        if self.handler_registry.task_count() >= 10:
            self._event_bus.emit_alert(
                "Task limit reached", "Maximum of 10 concurrent tasks allowed."
            )
            return

        handler = self.handler_registry.get_selected()
        if not handler or handler["tasks"]["Compute Graph Properties"]:
            return

        task = Task(
            task_id=str(uuid.uuid4()), task_type="Compute Graph Properties"
        )
        handler["tasks"]["Compute Graph Properties"] = task
        task.status = "Running"
        self._event_bus.emit_task_changed()
        logger.info(f"Task {task.task_id} created for graph properties computation")

        def on_finished(result, error=None):
            """Send signals to the main thread."""
            handler["tasks"]["Compute Graph Properties"] = None
            self._event_bus.emit_task_changed()
            if error:
                logger.error(f"Task {task.task_id} failed: {error}")
                self._event_bus.emit_alert("Failed to compute graph properties", repr(error))
                self._event_bus.emit_compute_properties_finished(False)
            else:
                self._event_bus.emit_handler_changed()
                self._event_bus.emit_compute_properties_finished(True)

        worker = Worker(
            NetworkService.compute_graph_properties,
            callback=on_finished,
            task=task,
            handler_registry=self.handler_registry,
            options=options,
        )
        task.thread = worker
        self._active_threads.append(worker)
        worker.finished.connect(lambda: self._active_threads.remove(worker))
        worker.start()

    def cancel_task(self, task_id: str):
        """Cancel a task."""
        for index in self.handler_registry.get_valid_indices():
            handler = self.handler_registry.get(index)
            if handler is None:
                continue
            tasks = handler["tasks"]
            for task_type, task in tasks.items():
                if task and task.task_id == task_id:
                    if task.status == "Running" and task.thread:
                        task.thread.terminate()
                        task.thread.wait()
                        handler["tasks"][task_type] = None
                        logger.info(
                            f"Task {task_id} cancelled at "
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self._event_bus.emit_task_changed()
                        return
                    if task.status == "Pending":
                        handler["tasks"][task_type] = None
                        logger.info(
                            f"Task {task_id} cancelled at "
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self._event_bus.emit_task_changed()
                        return
        logger.warning(f"Task {task_id} not found for cancellation")

    def cleanup_threads(self) -> None:
        """Clean up all active threads on app shutdown."""
        for thread in self._active_threads[:]:
            if thread.isRunning():
                thread.requestInterruption()
                if not thread.wait(1000):
                    thread.terminate()
                    thread.wait()
            self._active_threads.remove(thread)
