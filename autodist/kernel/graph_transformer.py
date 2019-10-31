"""Graph Transformer."""

from tensorflow.python.eager import context
from tensorflow.python.framework import device_spec, ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from autodist.graph_item import GraphItem
from autodist.kernel.device.resolver import DeviceResolver
from autodist.kernel.partitioner import VariablePartitioner
from autodist.kernel.replicator import Replicator
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.strategy.base import StrategyCompiler
from autodist.kernel.common import resource_variable
from autodist.kernel.common.utils import replica_prefix
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

    @property
    def num_local_replica(self):
        """Infer the nubmer of replica on this local machine."""
        replica_devices = \
            {device_spec.DeviceSpecV2.from_string(s) for s in self._strategy.graph_config.get('replicas')}
        return len({
            d for d in replica_devices
            if self._cluster.get_local_address() == self._cluster.get_address_from_task(d.job, d.task)
        })

    def transform(self, graph_item: GraphItem):
        """Call graph transformer to transform a graph item based on strategy and cluster."""
        with context.graph_mode():
            # Ensure the transformation happens under graph mode, no matter the outer mode is under eager or graph.

            visualization_util.log_graph(graph=graph_item.graph, name='original')

            graph_item, self._strategy.node_config = VariablePartitioner(self._strategy.node_config, graph_item)()

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

    def remap_io(self, graph_item, fetches, feed_dict=None):
        """
        Remap the user-provided fetches to the right list of fetches after graph transformations.

        It returns a list of new fetches that are necessary for distributed execution, e.g. in AllReduce,
        the train ops on all replicas need to be fetched in order to trigger the execution on all replicas.
        It also returns a remap_return_func that will be called in runner to map the actual fetched results (which
        are supposed be a superset of the original fetches from users code) back to user-desired results.

        Args:
            graph_item: the transformed graph item
            fetches: the original fetches from user code
            feed_dict: the original feed_dict from user code

        Returns:
            new_fetches (list) : a list of new fetches for execution
            remap_return_func (function): a function that maps the fetched results back to original fetches

        """
        if not isinstance(fetches, (tuple, list)):
            fetches = [fetches]

        def remap_return_func(returns):
            ret = [returns[name_to_remapped_indices[fetch.name]]
                   for fetch in fetches]
            return ret if len(ret) > 1 else ret[0]

        # TODO: Handle Feed Fetch for both Graph and FuncGraph in the style of tensorflow.python.client._HandleFetch
        new_fetches, name_to_remapped_indices = self._remap_fetches(graph_item.graph, fetches)
        new_feed_dict = self._remap_feed_dict(graph_item.graph, feed_dict)
        return new_fetches, new_feed_dict, remap_return_func

    def _remap_feed_dict(self, graph, feed_dict):
        if feed_dict is None:
            return None
        new_feed_dict = {}
        for t, v in feed_dict.items():
            try:
                d = {graph.get_tensor_by_name(t.name): v}
            except KeyError:
                # Temporary Workaround for SYM-9004
                d = {ops.prepend_name_scope(t.name, replica_prefix(i)): v for i in range(self.num_local_replica)}
                logging.warning('Feed key %s is remapped to all replicas for the same value.' % t.name)
            new_feed_dict.update(d)
        return new_feed_dict

    def _remap_fetches(self, graph, fetches):
        """
        Remap the fetches in graph following rules below.

        For fetches that are stateful operations (i.e. train_op), fetch them on all replicas.
        For fetches that are tensors or variables, only fetch it on master_replica.

        Args:
            graph (ops.Graph): The graph to be executed
            fetches (list): a list of fetches by users

        Returns:
            remapped_fetches (list): a list of fetches remapped
            name_to_remapped_indices (dict): a map from the name of the original fetch to its index in the new fetches

        """
        index = 0
        name_to_remapped_indices = {}
        remapped_fetches = []
        for fetch in fetches:
            remapped_fetch = self.__remap_fetch(graph, fetch)
            name_to_remapped_indices[fetch.name] = index
            remapped_fetches.extend(remapped_fetch)
            index += len(remapped_fetch)
        return remapped_fetches, name_to_remapped_indices

    def __remap_fetch(self, graph, fetch):
        remap = {
            ops.Tensor: graph.get_tensor_by_name,
            ops.Operation: graph.get_operation_by_name,
            ResourceVariable: lambda name: resource_variable.get_read_var_tensor(graph.get_tensor_by_name(name).op)
        }
        fetch_type = type(fetch)
        try:
            if fetch_type not in remap:
                raise TypeError('Fetch type {} not supported.'.format(fetch_type))
            return [remap[fetch_type](fetch.name)]
        except KeyError:
            master_replica_name = ops.prepend_name_scope(fetch.name, replica_prefix(0))
            # For fetches that are stateful operations (i.e. train_op), fetch them on all replicas
            # For fetches that are tensors or variables, only fetch it on master_replica
            if fetch_type is ops.Operation and remap[type(fetch)](master_replica_name).op_def.is_stateful:
                logging.warning('Fetch %s is remapped to all replicas' % fetch.name)
                return [remap[type(fetch)](ops.prepend_name_scope(fetch.name, replica_prefix(i)))
                        for i in range(self.num_local_replica)]
            else:
                logging.warning('Fetch %s is remapped to %s' % (fetch.name, master_replica_name))
                return [remap[type(fetch)](master_replica_name)]
