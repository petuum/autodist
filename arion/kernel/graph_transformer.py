"""Graph Transformer."""

from tensorflow.core.framework.attr_value_pb2 import AttrValue as pb2_AttrValue
from tensorflow.python.eager import context
from tensorflow.python.framework import device_spec

from autodist.graph_item import GraphItem
from autodist.kernel.partitioner import VariablePartitioner
from autodist.kernel.replicator import Replicator
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.utils import logging, visualization_util


class GraphTransformer:
    """Graph Transformer."""

    def __init__(self, compiled_strategy, cluster, graph_item):
        self._strategy = compiled_strategy
        self._cluster = cluster
        self.graph_item = graph_item

        # Set in _initialize_synchronizers
        self._num_local_replicas = 0
        self._num_workers = 0
        self._synchronizers = {}

    def transform(self):
        """Call graph transformer to transform a graph item based on strategy and cluster."""
        with context.graph_mode():
            graph_item = self.graph_item
            graph_item.prepare()
            # Ensure the transformation happens under graph mode, no matter the outer mode is under eager or graph.

            visualization_util.log_graph(graph=graph_item.graph, name='original')

            graph_item, self._strategy.node_config = VariablePartitioner.apply(self._strategy.node_config, graph_item)

            # Create Synchronizers for each node in the strategy
            self._initialize_synchronizers()

            # Replicate the graph (both in-graph and between-graph)
            new_graph_item = Replicator.apply(
                config=self._strategy.graph_config.replicas,
                cluster=self._cluster,
                graph_item=graph_item
            )

            # Apply synchronizers
            if self._num_local_replicas >= 1:
                new_graph_item = self._in_graph_apply(new_graph_item)
                logging.info('Successfully applied local in-graph replication')
                visualization_util.log_graph(new_graph_item.graph, 'after-in-graph')

            if self._num_workers >= 1:
                new_graph_item = self._between_graph_apply(new_graph_item)
                logging.info('Successfully applied between-graph replication')

            final_item = new_graph_item
            logging.info('Successfully built transformed graph')
            visualization_util.log_graph(graph=final_item.graph, name='transformed')

        return final_item

    def _initialize_synchronizers(self):
        self._synchronizers = {
            node.var_name: Synchronizer.create(node.WhichOneof('synchronizer'),
                                               getattr(node, node.WhichOneof('synchronizer')))
            for node in self._strategy.node_config
        }

        config = self._strategy.graph_config.replicas

        replica_devices = {device_spec.DeviceSpecV2.from_string(s) for s in config}
        replica_hosts = {self._cluster.get_address_from_task(d.job, d.task) for d in replica_devices}
        self._num_workers = len(replica_hosts)

        local_canonical_replica_devices = sorted({
            d.to_string() for d in replica_devices
            if self._cluster.get_local_address() == self._cluster.get_address_from_task(d.job, d.task)
        })
        logging.debug('Local replica devices: {}'.format(local_canonical_replica_devices))
        self._num_local_replicas = len(local_canonical_replica_devices)

        local_worker_id = self._cluster.get_local_worker_task_index()
        local_worker_device = '/job:worker/task:{}'.format(local_worker_id)

        for synchronizer in self._synchronizers.values():
            synchronizer.assign_cluster_information(self._num_workers, self._num_local_replicas,
                                                    local_worker_device, local_worker_id,
                                                    sorted({d.to_string() for d in replica_devices}),
                                                    is_chief=self._cluster.is_chief())

    @property
    def num_local_replicas(self):
        """Return the number of local replicas."""
        assert self._num_local_replicas != 0  # ensure initialized
        return self._num_local_replicas

    def _in_graph_apply(self, graph_item: GraphItem):
        """
        Perform in-graph synchronization of the graph.

        Args:
            graph_item (GraphItem): The graph to replication.

        Returns:
            GraphItem
        """
        new_graph_item = graph_item
        for var_name, syncer in self._synchronizers.items():
            new_graph_item = syncer.in_graph_apply(new_graph_item, var_name)
        return new_graph_item

    def _between_graph_apply(self, multi_gpu_graph_item: GraphItem):
        """
        Perform between-graph replication of the graph.

        Args:
            multi_gpu_graph_item (GraphItem): The graph to replication.

        Returns:
            GraphItem
        """
        new_graph_item = multi_gpu_graph_item
        for var_name, syncer in self._synchronizers.items():
            new_graph_item = syncer.between_graph_apply(new_graph_item, var_name)
        self._prune_colocation_groups(new_graph_item)
        # TODO: make this work
        # update_shard_values_for_worker(num_workers, worker_id)
        return new_graph_item

    # TODO(Hao): this seems still problematic
    @staticmethod
    def _prune_colocation_groups(graph_item):
        for op in graph_item.graph.get_operations():
            # Now prune the graph to have the right colocation constraints
            colocation_groups = [(c, graph_item.get_colocation_op(c)) for c in op.colocation_groups()]
            # We don't want any colocation groups that are just this `op`
            colocation_groups = [(c, bind_op) for (c, bind_op) in colocation_groups if bind_op != op]
            if colocation_groups:
                device_to_bind_to = colocation_groups[-1][1].device
                new_colocation_groups = [c for (c, op) in colocation_groups if op.device == device_to_bind_to]
                op._set_device(device_to_bind_to)
                op._set_attr("_class", pb2_AttrValue(list=pb2_AttrValue.ListValue(s=new_colocation_groups)))
            else:
                try:
                    if op.get_attr("_class"):
                        op._clear_attr("_class")
                except ValueError:
                    pass