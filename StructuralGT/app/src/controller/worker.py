"""Worker thread for StructuralGT GUI."""

import logging
from collections.abc import Callable
from datetime import datetime

from model.task_list_model import Task
from PySide6.QtCore import QThread

logger = logging.getLogger(__name__)

class Worker(QThread):
    """Worker thread for StructuralGT GUI."""

    def __init__(
        self,
        func: Callable,
        callback: Callable | None = None,
        task: Task | None = None,
        *args: object,
        **kwargs: object,
    ):
        """Initialize the worker."""
        super().__init__()
        self.func = func
        self.callback = callback
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Run the worker."""
        try:
            if self.task:
                logger.info(
                    f"Task {self.task.task_id} started at "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                result = self.func(*self.args, **self.kwargs)

                if self.callback:
                    self.callback(result)
                    logger.info(
                        f"Task {self.task.task_id} completed at "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
        except Exception as e:
            if self.callback:
                self.callback(None, e)
