"""Central event dispatcher for StructuralGT GUI."""

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class EventBus(QObject):
    """Central event dispatcher using publich-subscribe pattern."""

    _instance: Optional["EventBus"] = None

    view_changed = Signal(str)
    alert = Signal(str, str)
    handler_changed = Signal()
    binarize_finished = Signal(bool)
    extract_graph_finished = Signal(str)
    compute_properties_finished = Signal(bool)
    task_changed = Signal()
    network_added = Signal(int)
    network_deleted = Signal(int)
    network_selected = Signal(int)

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get the singleton EventBus instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance  # type: ignore[return-value]

    def __init__(self) -> None:
        """Initialize the singleton EventBus instance."""
        if hasattr(self, "_initialized"):
            return
        super().__init__()
        self._initialized = True
        logger.debug("EventBus initialized")

    def emit_view_changed(self, view_name: str) -> None:
        """Emit the view_changed signal."""
        logger.debug(f"Emitting view_changed: {view_name}")
        self.view_changed.emit(view_name)

    def emit_alert(self, title: str, message: str) -> None:
        """Emit the alert signal."""
        logger.debug(f"Emitting alert: {title}")
        self.alert.emit(title, message)

    def emit_handler_changed(self) -> None:
        """Emit the handler_changed signal."""
        logger.debug("Emitting handler_changed")
        self.handler_changed.emit()

    def emit_binarize_finished(self, success: bool) -> None:
        """Emit the binarize_finished signal."""
        logger.debug(f"Emitting binarize_finished: {success}")
        self.binarize_finished.emit(success)

    def emit_extract_graph_finished(self, pipeline: str) -> None:
        """Emit the extract_graph_finished signal."""
        logger.debug(f"Emitting extract_graph_finished: {pipeline}")
        self.extract_graph_finished.emit(pipeline)

    def emit_compute_properties_finished(self, success: bool) -> None:
        """Emit the compute_properties_finished signal."""
        logger.debug(f"Emitting compute_properties_finished: {success}")
        self.compute_properties_finished.emit(success)

    def emit_task_changed(self) -> None:
        """Emit the task_changed signal."""
        logger.debug("Emitting task_changed")
        self.task_changed.emit()

    def emit_network_added(self, handler_index: int) -> None:
        """Emit the network_added signal."""
        logger.debug(f"Emitting network_added: {handler_index}")
        self.network_added.emit(handler_index)

    def emit_network_deleted(self, handler_index: int) -> None:
        """Emit the network_deleted signal."""
        logger.debug(f"Emitting network_deleted: {handler_index}")
        self.network_deleted.emit(handler_index)

    def emit_network_selected(self, handler_index: int) -> None:
        """Emit the network_selected signal."""
        logger.debug(f"Emitting network_selected: {handler_index}")
        self.network_selected.emit(handler_index)
