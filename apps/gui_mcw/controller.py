import re
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tempfile
import shutil
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport
from ovito.gui import create_qwidget
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    # False at run time, only for a type-checker
    from _typeshed import SupportsWrite

from .handler import NetworkHandler, PointNetworkHandler
from .table_model import TableModel
from .list_model import ListModel
from .qthread_worker import QThreadWorker, WorkerTask

from StructuralGT import __version__

ALLOWED_IMG_EXTENSIONS = ["*.jpg", "*.jpeg", "*.tif", "*.tiff"]

class MainController(QObject):
    """Exposes a method to refresh the image in QML"""

    # updateProgressSignal = Signal(int, str)
    # taskTerminatedSignal = Signal(bool, list)
    # projectOpenedSignal = Signal(str)
    # enableRectangularSelectionSignal = Signal(bool)
    # showCroppingToolSignal = Signal(bool)
    # showUnCroppingToolSignal = Signal(bool)
    # performCroppingSignal = Signal(bool)

    showAlertSignal = Signal(str, str)
    errorSignal = Signal(str)
    changeImageSignal = Signal()
    imageChangedSignal = Signal()
    taskFinishedSignal = Signal(bool, object)  # success/fail (True/False), result (object)

    def __init__(self, qml_app: QApplication):
        super().__init__()
        self.qml_app = qml_app

        # Create network objects
        self.handlers = []
        self.selected_index = 0

        # Create QThreadWorker for long tasks
        self.wait_flag = False
        self.worker_task = WorkerTask()
        self.thread = QThreadWorker(None, None)

        # Create Models
        self.imagePropsModel = TableModel([])
        self.graphPropsModel = TableModel([])
        self.imageListModel = ListModel([])

    @Slot(str, result=str)
    def get_file_extensions(self, option):
        if option == "img":
            pattern_string = " ".join(ALLOWED_IMG_EXTENSIONS)
            return f"Image files ({pattern_string})"
        elif option == "proj":
            return "Project files (*.sgtproj)"
        elif option == "csv":
            return "CSV files (*.csv)"
        else:
            return ""

    def verify_path(self, a_path):
        if not a_path:
            logging.info(
                "No folder/file selected.", extra={"user": "SGT Logs"}
            )
            self.showAlertSignal.emit(
                "File/Directory Error", "No folder/file selected."
            )
            return False

        # Convert QML "file:///" path format to a proper OS path
        if a_path.startswith("file:///"):
            if sys.platform.startswith("win"):
                # Windows Fix (remove extra '/')
                a_path = a_path[8:]
            else:
                # macOS/Linux (remove "file://")
                a_path = a_path[7:]

        # Normalize the path
        a_path = os.path.normpath(a_path)

        if not os.path.exists(a_path):
            logging.exception(
                "Path Error: %s", IOError, extra={"user": "SGT Logs"}
            )
            self.showAlertSignal.emit(
                "Path Error",
                f"File/Folder in {a_path} does not exist. Try again.",
            )
            return False
        return a_path

    def get_selected_handler(self):
        # TODO: modify the error message
        try:
            return self.handlers[self.selected_index]
        except IndexError:
            logging.info(
                "No Image Error: Please import/add an image.",
                extra={"user": "SGT Logs"},
            )
            self.showAlertSignal.emit(
                "No Image Error", "No image added! Please import/add an image."
            )
            return None

    # TODO: do we need this?
    @Slot(result=bool)
    def img_loaded(self):
        if self.handlers:
            return self.get_selected_handler().img_loaded
        return False

    @Slot(result=bool)
    def graph_loaded(self):
        if self.handlers:
            return self.get_selected_handler().graph_loaded
        return False

    # TODO: do we need this?
    @Slot(result=bool)
    def is_3d(self):
        return self.get_selected_handler().is_3d

    @Slot(result=str)
    def display_type(self):
        if not self.handlers:
            return ""
        return self.get_selected_handler().display_type

    # TODO: point network does not have this
    @Slot(result=int)
    def get_selected_slice_index(self):
        """Get the selected slice index of the selected image."""
        if not self.handlers:
            return -1
        return self.get_selected_handler().selected_slice_index

    # TODO: point network does not have this
    @Slot(result=int)
    def get_number_of_slices(self):
        """Get the number of slices of the selected image."""
        return self.get_selected_handler().network.image.shape[0]

    # TODO: point network does not have this
    @Slot(int, result=bool)
    def set_selected_slice_index(self, index):
        """Set the selected slice index of the selected image."""
        image = self.get_selected_handler()
        if image and 0 <= index < image.network.image.shape[0]:
            image.selected_slice_index = index
            self.load_image()
            return True
        return False

    @Slot(result=bool)
    def load_prev_slice(self):
        """Load the previous slice of the selected image."""
        handler = self.get_selected_handler()
        if handler and handler.selected_slice_index > 0:
            handler.selected_slice_index -= 1
            self.load_image()
            return True
        return False

    @Slot(result=bool)
    def load_next_slice(self):
        """Load the next slice of the selected image."""
        handler = self.get_selected_handler()
        if handler and handler.selected_slice_index < handler.network.image.shape[0] - 1:
            handler.selected_slice_index += 1
            self.load_image()
            return True
        return False

    # TODO: make it compatible with point network
    def create_handler(self, img_path, type):
        """
        A function that processes a selected image file and creates an analyzer object with default configurations.

        Args:
            img_path: file path to image

        Returns:
        """

        img_path = self.verify_path(img_path)
        if not img_path:
            return False
        print(img_path)

        # Try reading the image
        try:
            if type == "3D":
                # Create a 3D image object
                self.handlers.append(NetworkHandler(img_path, dim=3))
            elif type == "2D":
                # Create a temporary directory for the image
                prefix = "sgt_"
                temp_dir = tempfile.TemporaryDirectory(prefix=prefix)

                # Copy the image to the temporary directory
                temp_img_path = os.path.join(
                    temp_dir.name, os.path.basename(img_path)
                )
                shutil.copy(img_path, temp_img_path)

                print(temp_dir.name)

                # Create a 2d image object
                self.handlers.append(NetworkHandler(temp_dir, dim=2))
            elif type == "Point":
                positions = pd.read_csv(img_path)
                positions = positions[["x", "y", "z"]].values
                self.handlers.append(PointNetworkHandler(positions, cutoff=1.0))

            return True

        except Exception as err:
            logging.exception(
                "File Error: %s", err, extra={"user": "SGT Logs"}
            )
            self.showAlertSignal.emit(
                "File Error", "Error processing image. Try again."
            )
            return False

    # TODO: make it compatible with point network
    @Slot(str, str, result=bool)
    def add_handler(self, img_path, type):
        """Verify and validate an image path, use it to create an Network and load it in view."""
        is_created = self.create_handler(img_path, type)
        if is_created:
            
            self.imageListModel.reset_data(
                [{"id": i, "name": str(handler.path)}
                for i, handler in enumerate(self.handlers)]
            )

            self.load_image()
            return True
        return False

    @Slot(int, result=bool)
    def delete_handler(self, index):
        """Delete an image from the list of images."""
        # Remove the image from the list
        del self.handlers[index]
        self.selected_index = 0 if not self.handlers else min(
            self.selected_index, len(self.handlers) - 1
        )

        # Reset the models
        self.imageListModel.reset_data(
            [{"id": i, "name": str(handler.path)}
                for i, handler in enumerate(self.handlers)]
        )

        # Load the first image or reset if no images left
        self.load_image()
        return True

    # TODO: make it compatible with point network
    @Slot(int)
    def load_image(self, index=None):
        """Load the image of the selected network handler."""
        print("Loading image...")
        try:
            if index is not None:
                if index == self.selected_index:
                    return
                else:
                    self.selected_index = index

            self.changeImageSignal.emit()
        except Exception as err:
            # self.delete_sgt_object()
            self.selected_index = 0
            logging.exception(
                "Image Loading Error: %s", err, extra={"user": "SGT Logs"}
            )
            self.showAlertSignal.emit(
                "Image Error", "Error loading image. Try again."
            )

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        curr_img_view = np.random.randint(0, 4)
        unique_num = (
            self.selected_index
            + curr_img_view
            + np.random.randint(low=21, high=1000)
        )
        return "image://imageProvider/" + str(unique_num)

    # TODO: point network does not have this
    @Slot(str)
    def run_binarizer(self, options):
        """Run the binarizer on the selected image"""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        image = self.get_selected_handler()
        if image is None:
            self.wait_flag = False
            return

        self.wait_flag = True
        if options:
            image.options = json.loads(options)

        self.thread = QThreadWorker(
            self.worker_task.task_run_binarizer,
            (image,),
        )

        self.worker_task.taskFinishedSignal.connect(self._on_binarizer_finished)
        self.worker_task.taskFinishedSignal.connect(self.thread.quit)

        self.thread.start()

    @Slot(bool, object)
    def _on_binarizer_finished(self, success, result):
        self.wait_flag = False
        if success:
            self.toggle_current_img_view("binary")
        else:
            self.showAlertSignal.emit("Binarizer Error", "Error applying binarizer.")

    # TODO: point network does not have this
    @Slot()
    def run_graph_extraction(self):
        """Run the graph extraction on the selected image"""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        image = self.get_selected_handler()
        if image is None:
            self.wait_flag = False
            return

        self.wait_flag = True
        self.thread = QThreadWorker(
            self.worker_task.task_run_graph_extraction,
            (image,),
        )
        self.worker_task.taskFinishedSignal.connect(self._on_graph_extraction_finished)
        self.worker_task.taskFinishedSignal.connect(self.thread.quit)
        self.thread.start()

    @Slot(bool, object)
    def _on_graph_extraction_finished(self, success, result):
        self.wait_flag = False
        if success:
            self.toggle_current_img_view("graph")
        else:
            self.showAlertSignal.emit(result[0], result[1])

    @Slot(str, result=bool)
    def toggle_current_img_view(self, display_type):
        print(f"Display type: {display_type}")

        image = self.get_selected_handler()

        if display_type == "binary" and not image.binary_loaded:
            # Use the default options
            self.run_binarizer(options=None)
            return False
        elif display_type == "graph" and not image.graph_loaded:
            self.run_graph_extraction()
            return False
        elif image.display_type == display_type:
            return True
        else:
            image.display_type = display_type
            self.load_image()
            return True

    def load_graph(self):
        """Render and visualize OVITO graph network simulation."""
        try:
            # Clear any existing scene
            for p_line in list(scene.pipelines):
                p_line.remove_from_scene()

            # Create OVITO data pipeline
            image = self.get_selected_handler()
            gsd_file = os.path.join(
                image.img_path.name, "Binarized/network.gsd"
            ) if not image.is_3d else os.path.join(
                image.img_path, "Binarized/network.gsd"
            )
            pipeline = import_file(gsd_file)
            pipeline.add_to_scene()

            vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))
            ovito_widget = create_qwidget(vp, parent=self.qml_app.activeWindow())
            ovito_widget.setMinimumSize(800, 500)
            vp.zoom_all((800, 500))
            ovito_widget.show()

        except Exception as e:
            print("Graph Simulation Error:", e)

    @Slot()
    def update_image_model(self):
        """Update the image properties model with the selected image's properties."""
        image = self.get_selected_handler()

        if image is None:
            return

        if image.is_3d:
            image_props = [
                ["Name", f"{image.img_path}"],
                ["Depth x Height x Width", f"{image.network.image.shape[0]} x {image.network.image.shape[1]} x {image.network.image.shape[2]}"],
                ["Dimensions", "3D"]
            ]
        else:
            image_props = [
                ["Name", f"{image.img_path.name}"],
                ["Height x Width", f"{image.network.image.shape[0]} x {image.network.image.shape[1]}"],
                ["Dimensions", "2D"]
            ]

        self.imagePropsModel.reset_data(image_props)

    @Slot()
    def update_graph_model(self):
        """Update the graph properties model with the selected image's graph properties."""
        image = self.get_selected_handler()

        if image is None:
            return

        if image.graph_loaded:
            graph_props = [
                ["Edge Count", f"{image.network.graph.ecount()}"],
                ["Node Count", f"{image.network.graph.vcount()}"],
            ]

            for key, value in image.properties.items():
                if value:
                    graph_props.append([key, f"{value:.5f}"])

            self.graphPropsModel.reset_data(graph_props)

    @Slot(str)
    def run_graph_analysis(self, options):
        """Compute selected graph properties of the selected image."""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        image = self.get_selected_handler()
        if image is None:
            self.wait_flag = False
            return

        self.wait_flag = True
        options = json.loads(options)

        self.thread = QThreadWorker(
            self.worker_task.task_run_graph_analysis,
            (image,options),
        )

        self.worker_task.taskFinishedSignal.connect(self._on_graph_analysis_finished)
        self.worker_task.taskFinishedSignal.connect(self.thread.quit)

        self.thread.start()

    @Slot(bool, object)
    def _on_graph_analysis_finished(self, success, result):
        self.wait_flag = False
        if success:
            self.update_graph_model()
        else:
            self.showAlertSignal.emit("Graph Analysis Failed", "Error computing graph properties.")

    @Slot(result=str)
    def get_sgt_version(self):
        """"""
        # Copyright (C) 2024, the Regents of the University of Michigan.
        return f"StructuralGT v{__version__}"

    @Slot(result=str)
    def get_about_details(self):
        about_app = (
            f"A software tool that allows graph theory analysis of nano-structures. This is a modified version "
            "of StructuralGT initially proposed by Drew A. Vecchio, DOI: "
            "<html><a href='https://pubs.acs.org/doi/10.1021/acsnano.1c04711'>10.1021/acsnano.1c04711</a></html><html><br/><br/></html>"
            "Contributors:<html><br/></html>"
            "1. Nicolas Kotov<html><br/></html>"
            "2. Dickson Owuor<html><br/></html>"
            "3. Alain Kadar<html><br/><br/></html>"
            "Documentation:<html><br/></html>"
            "<html><a href='https://structuralgt.readthedocs.io'>structuralgt.readthedocs.io</a></html><html><br/><br/></html>"
            f"{self.get_sgt_version()}<html><br/></html>"
            "Copyright (C) 2018-2025, The Regents of the University of Michigan.<html><br/></html>"
            "License: GPL GNU v3<html><br/></html>"
        )
        return about_app
