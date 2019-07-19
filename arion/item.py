"""Item as metagraph wrapper."""

from tensorflow.python.framework import ops
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.python.training.saver import export_meta_graph, import_meta_graph


class Item:
    """Item as metagraph wrapper."""

    def __init__(self, graph):
        self._metagraphdef = export_meta_graph(graph=graph)
        self._graph = ops.Graph()
        with self._graph.as_default():
            import_meta_graph(self._metagraphdef)
        self._feeds = {}
        self._fetchs = {}

    def get_variables_to_sync(self):
        """Get variables that need to be synchronized if doing data parallelism."""
        with self._graph.as_default():
            return trainable_variables()
