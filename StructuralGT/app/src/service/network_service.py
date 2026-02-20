"""Network service for StructuralGT GUI."""

from model.handler import HandlerRegistry, NetworkHandler, PointNetworkHandler

from StructuralGT.betweenness import NodeBetweenness
from StructuralGT.electronic import Electronic
from StructuralGT.geometric import Nematic
from StructuralGT.structural import (
    Assortativity,
    Closeness,
    Clustering,
    Degree,
    Size,
)


class NetworkService:
    """Network service for StructuralGT GUI."""

    @staticmethod
    def add_network(handler_registry: HandlerRegistry, folder_path: str, dim: int):
        """Add a NetworkHandler to the handler registry."""
        network_handler = NetworkHandler(folder_path, dim=dim)
        handler_registry.add(network_handler)
        return

    @staticmethod
    def add_point_network(handler_registry: HandlerRegistry, file_path: str):
        """Add a PointNetworkHandler to the handler registry."""
        point_handler = PointNetworkHandler(file_path)
        handler_registry.add(point_handler)
        return

    @staticmethod
    def delete_network(handler_registry: HandlerRegistry, index: int):
        """Delete a Handler from the handler registry."""
        handler_registry.delete(index)
        return

    @staticmethod
    def binarize_selected_network(
        handler_registry: HandlerRegistry,
        options: dict,
    ):
        """Binarize the selected network."""
        handler = handler_registry.get_selected()
        if handler and isinstance(handler, NetworkHandler):
            handler["binarize_options"] = options
            handler["network"].binarize(options)
            handler["ui_properties"]["binarized_loaded"] = True
        return

    @staticmethod
    def extract_graph_from_selected_network(
        handler_registry: HandlerRegistry,
        weight_type,
    ):
        """Extract the graph from the selected network."""
        handler = handler_registry.get_selected()
        if handler and isinstance(handler, NetworkHandler):
            handler["network"].img_to_skel()
            handler["network"].set_graph(weight_type=weight_type)
            handler["ui_properties"]["extracted_loaded"] = True
            handler["network_properties"]["weight_type"] = weight_type
        return

    @staticmethod
    def compute_size(handler_registry: HandlerRegistry):
        """Compute the size object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            size = Size()
            size.compute(handler["network"])
            return size
        return None

    @staticmethod
    def compute_clustering(handler_registry: HandlerRegistry):
        """Compute the clustering object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            clustering = Clustering()
            clustering.compute(handler["network"])
            return clustering
        return None

    @staticmethod
    def compute_assortativity(handler_registry: HandlerRegistry):
        """Compute the assortativity object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            assortativity = Assortativity()
            assortativity.compute(handler["network"])
            return assortativity
        return None

    @staticmethod
    def compute_closeness(handler_registry: HandlerRegistry):
        """Compute the closeness object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            closeness = Closeness()
            closeness.compute(handler["network"])
            return closeness
        return None

    @staticmethod
    def compute_degree(handler_registry: HandlerRegistry):
        """Compute the degree object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            degree = Degree()
            degree.compute(handler["network"])
            return degree
        return None

    @staticmethod
    def compute_nematic(handler_registry: HandlerRegistry):
        """Compute the nematic object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            nematic = Nematic()
            nematic.compute(handler["network"])
            return nematic
        return None

    @staticmethod
    def compute_betweenness_centrality(handler_registry: HandlerRegistry):
        """Compute the nodebetweenness object of the selected network."""
        handler = handler_registry.get_selected()
        if handler:
            betweenness = NodeBetweenness()
            betweenness.compute(handler["network"])
            return betweenness
        return None

    @staticmethod
    def compute_graph_properties(
        handler_registry: HandlerRegistry,
        options: dict,
    ):
        """Compute the graph properties of the selected network."""
        handler = handler_registry.get_selected()
        if handler is not None:
            graph_properties = handler["network_properties"].copy()
            if options["diameter"] or options["density"]:
                if handler["property_cache"]["Size"] is None:
                    handler["property_cache"]["Size"] = NetworkService.compute_size(
                        handler_registry
                    )
                if options["diameter"]:
                    graph_properties["diameter"] = handler["property_cache"][
                        "Size"
                    ].diameter
                if options["density"]:
                    graph_properties["density"] = handler["property_cache"][
                        "Size"
                    ].density
            if options["average_clustering_coefficient"]:
                if handler["property_cache"]["Clustering"] is None:
                    handler["property_cache"]["Clustering"] = (
                        NetworkService.compute_clustering(handler_registry)
                    )
                graph_properties["average_clustering_coefficient"] = handler[
                    "property_cache"
                ]["Clustering"].average_clustering_coefficient
            if options["assortativity"]:
                if handler["property_cache"]["Assortativity"] is None:
                    handler["property_cache"]["Assortativity"] = (
                        NetworkService.compute_assortativity(handler_registry)
                    )
                graph_properties["assortativity"] = handler["property_cache"][
                    "Assortativity"
                ].assortativity
            if options["average_closeness"]:
                if handler["property_cache"]["Closeness"] is None:
                    handler["property_cache"]["Closeness"] = (
                        NetworkService.compute_closeness(handler_registry)
                    )
                graph_properties["average_closeness"] = handler["property_cache"][
                    "Closeness"
                ].average_closeness
            if options["average_degree"]:
                if handler["property_cache"]["Degree"] is None:
                    handler["property_cache"]["Degree"] = NetworkService.compute_degree(
                        handler_registry
                    )
                graph_properties["average_degree"] = handler["property_cache"][
                    "Degree"
                ].average_degree
            if options["nematic_order_parameter"]:
                if handler["property_cache"]["Nematic"] is None:
                    handler["property_cache"]["Nematic"] = (
                        NetworkService.compute_nematic(handler_registry)
                    )
                graph_properties["nematic_order_parameter"] = handler["property_cache"][
                    "Nematic"
                ].nematic_order_parameter
            if options["average_betweenness_centrality"]:
                if handler["property_cache"]["NodeBetweenness"] is None:
                    handler["property_cache"]["NodeBetweenness"] = (
                        NetworkService.compute_betweenness_centrality(handler_registry)
                    )
                graph_properties["average_betweenness_centrality"] = handler[
                    "property_cache"
                ]["NodeBetweenness"].average_node_betweenness
            if options["effective_resistance"]:
                pass
            handler["network_properties"] = graph_properties
            return True
        return False

    @staticmethod
    def plot_skeleton(handler_registry: HandlerRegistry, ax):
        """Plot the skeleton of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        ax.imshow(handler["network"].skeleton, cmap="gray")
        ax.axis("off")
        ax.figure.tight_layout()

    @staticmethod
    def plot_graph(handler_registry: HandlerRegistry, ax):
        """Plot the graph of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        handler["network"].graph_plot(ax=ax)
        ax.figure.tight_layout()

    @staticmethod
    def plot_degree_heatmap(handler_registry: HandlerRegistry, ax):
        """Plot the degree heatmap of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        degree = handler["property_cache"]["Degree"]
        ax.set_title("Degree Heatmap", fontsize=10)
        handler["network"].node_plot(parameter=degree.degree, ax=ax)
        ax.figure.tight_layout()

    @staticmethod
    def plot_betweenness_centrality_heatmap(handler_registry: HandlerRegistry, ax):
        """Plot the betweenness centrality heatmap of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        betweenness = handler["property_cache"]["NodeBetweenness"]
        ax.set_title("Betweenness Centrality Heatmap", fontsize=10)
        handler["network"].node_plot(parameter=betweenness.node_betweenness, ax=ax)
        ax.figure.tight_layout()

    @staticmethod
    def plot_closeness_centrality_heatmap(handler_registry: HandlerRegistry, ax):
        """Plot the closeness centrality heatmap of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        closeness = handler["property_cache"]["Closeness"]
        ax.set_title("Closeness Centrality Heatmap", fontsize=10)
        handler["network"].node_plot(parameter=closeness.closeness, ax=ax)
        ax.figure.tight_layout()

    @staticmethod
    def plot_degree_distribution(handler_registry: HandlerRegistry, ax):
        """Plot the degree distribution of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        degree = handler["property_cache"]["Degree"]
        ax.set_title("Degree Distribution", fontsize=10)
        ax.hist(degree.degree, density=False, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Degree value")
        ax.set_ylabel("Count")
        ax.figure.tight_layout()

    @staticmethod
    def plot_betweenness_centrality_distribution(handler_registry: HandlerRegistry, ax):
        """Plot the betweenness centrality distribution of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        betweenness = handler["property_cache"]["NodeBetweenness"]
        ax.set_title("Betweenness Centrality Distribution", fontsize=10)
        ax.hist(
            betweenness.node_betweenness,
            density=False,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Betweenness value")
        ax.set_ylabel("Count")
        ax.figure.tight_layout()

    @staticmethod
    def plot_closeness_centrality_distribution(handler_registry: HandlerRegistry, ax):
        """Plot the closeness centrality distribution of the selected network."""
        handler = handler_registry.get_selected()
        if handler is None or not isinstance(handler, NetworkHandler):
            return
        if handler["network_properties"].get("dim") != 2:
            return
        closeness = handler["property_cache"]["Closeness"]
        ax.set_title("Closeness Centrality Distribution", fontsize=10)
        ax.hist(closeness.closeness, density=False, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Closeness value")
        ax.set_ylabel("Count")
        ax.figure.tight_layout()
