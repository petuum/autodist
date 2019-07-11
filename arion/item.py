"""Item as metagraph wrapper."""

from tensorflow.python.training.saver import export_meta_graph


class Item:
    """Item as metagraph wrapper."""

    def __init__(self, graph):
        self._graph = graph
        self._metagraphdef = export_meta_graph(graph=graph)
        self._feeds = {}
        self._fetchs = {}

    def get_variables(self):
        """Get variables."""
        pass
