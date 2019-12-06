"""Feed and Fetch Remapper."""

from tensorflow.python.framework import device_spec, ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from autodist.kernel.common.resource_variable_utils import get_read_var_tensor
from autodist.kernel.common.utils import replica_prefix
from autodist.utils import logging


class Remapper:
    """Feed and Fetch Remapper."""

    def __init__(self, compiled_strategy, cluster):
        self._strategy = compiled_strategy
        self._cluster = cluster

    @property
    def num_local_replica(self):
        """Infer the nubmer of replica on this local machine."""
        replica_devices = \
            {device_spec.DeviceSpecV2.from_string(s) for s in self._strategy.graph_config.replicas}
        return len({
            d for d in replica_devices
            if self._cluster.get_local_address() == self._cluster.get_address_from_task(d.job, d.task)
        })

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
            ResourceVariable: lambda name: get_read_var_tensor(graph.get_tensor_by_name(name).op)
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
                # For Debugging:
                # return [remap[type(fetch)](ops.prepend_name_scope(fetch.name, replica_prefix(i)))
                #        for i in range(self.num_local_replica)]
