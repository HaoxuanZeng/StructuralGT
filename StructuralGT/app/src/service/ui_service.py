"""UI service for StructuralGT GUI."""

import pathlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from model.handler import HandlerRegistry, NetworkHandler, PointNetworkHandler


class UIService:
    """UI service for StructuralGT GUI."""

    @staticmethod
    def set_selected_index(handler_registry: HandlerRegistry, index: int):
        """Select the handler at the given index."""
        handler_registry.select(index)
        return

    @staticmethod
    def set_selected_slice_index(handler_registry: HandlerRegistry, index: int):
        """Set the selected handler at the given slice index."""
        handler = handler_registry.get_selected()
        if handler and isinstance(handler, NetworkHandler):
            handler["ui_properties"]["selected_slice_index"] = index
        return

    @staticmethod
    def get_selected_slice_raw_image(
        handler_registry: HandlerRegistry, index: int
    ) -> Optional[np.ndarray]:
        """Get the selected slice raw image from the selected handler."""
        handler = handler_registry.get_selected()
        image = None
        if handler and isinstance(handler, NetworkHandler):
            dim = handler["network_properties"].get("dim", None)
            if dim == 2:
                image = handler["network"].image
            elif dim == 3:
                image = handler["network"].image[index, :, :]
        return image

    @staticmethod
    def get_selected_slice_binarized_image(
        handler_registry: HandlerRegistry, index: int
    ) -> Optional[np.ndarray]:
        """Get the selected slice binarized image from the selected handler."""
        handler = handler_registry.get_selected()
        image = None
        if handler and isinstance(handler, NetworkHandler):
            input_dir = handler["paths"]["input_dir"]
            dim = handler["network_properties"].get("dim", None)
            index = (
                index + 1 if dim == 3 else index
            )  # FIXME: This is restricted by StructuralGT library
            binarized_filename = f"slice{str(index).zfill(4)}.tiff"
            image_path = pathlib.Path(input_dir) / "Binarized" / binarized_filename
            image = cv2.imread(str(image_path))
        return image

    @staticmethod
    def get_selected_extracted_graph(
        handler_registry: HandlerRegistry,
    ) -> Optional[str]:
        """Get the selected extracted graph from the selected handler."""
        handler = handler_registry.get_selected()
        gsd_file = None
        if handler:
            input_dir = handler["paths"]["input_dir"]
            if isinstance(handler, NetworkHandler):
                gsd_file = pathlib.Path(input_dir) / "Binarized" / "network.gsd"
            elif isinstance(handler, PointNetworkHandler):
                gsd_file = pathlib.Path(input_dir).parent / "skel.gsd"
        return str(gsd_file)
