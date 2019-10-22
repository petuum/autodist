"""Graph Transformer."""

from tensorflow.python.eager import context

from autodist.graph_item import GraphItem
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.replication.replicator import Replicator
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.strategy.base import StrategyCompiler
from autodist.utils import logging, visualization_util


class GraphTransformer:
    """Graph Transformer."""

    def __init__(self, strategy, cluster):
        self._cluster = cluster

        # Prepare a compiled strategy for graph transformation.
        logging.info('Raw strategy: %s' % strategy)
        device_resolver = DeviceResolver(self._cluster)
        self._strategy = StrategyCompiler().set_device_resolver(device_resolver.resolve_to_device_str). \
            compile(strategy)
        logging.info('Compiled strategy: %s' % self._strategy)

    def __call__(self, graph_item: GraphItem):
        """Call graph transformer to transform a graph item based on strategy and cluster."""
        with context.graph_mode():
            # Ensure the transformation happens under graph mode, no matter the outer mode is under eager or graph.

            visualization_util.log_graph(graph=graph_item.graph, name='original')

            # Create Synchronizers for each node in the strategy
            synchronizers = {
                name: Synchronizer.create(node['synchronizer']['type'], **node['synchronizer']['config'])
                for name, node in self._strategy.node_config.items()
            }

            # Replicate the graph (both in-graph and between-graph)
            r = Replicator(
                config=self._strategy.graph_config.get('replicas'),
                cluster=self._cluster,
                synchronizers=synchronizers
            )

            # TODO: lift r.apply logic out to this scope
            final_item = r.apply(graph_item)
            logging.info('Successfully built transformed graph')
            visualization_util.log_graph(graph=final_item.graph, name='transformed')

        return final_item
