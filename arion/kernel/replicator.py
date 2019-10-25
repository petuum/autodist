"""Replicator."""

from tensorflow.core.framework.attr_value_pb2 import AttrValue as pb2_AttrValue
from tensorflow.python import ops, import_graph_def
from tensorflow.python.framework import device_spec, kernels
from tensorflow.python.framework.device_spec import DeviceSpecV2
from tensorflow.python.ops.resource_variable_ops import _from_proto_fn

from autodist.graph_item import GraphItem
from autodist.kernel.common import resource_variable
from autodist.kernel.common.utils import replica_prefix, strip_replica_prefix
from autodist.utils import logging


class Replicator:
    """Replicator."""

    def __init__(self, config, cluster, synchronizers):
        self._cluster = cluster
        self._synchronizers = synchronizers

        self._replica_devices = {device_spec.DeviceSpecV2.from_string(s) for s in config}
        self._replica_hosts = {cluster.get_address_from_task(d.job, d.task) for d in self._replica_devices}
        self._num_workers = len(self._replica_hosts)
        self._local_canonical_replica_devices = sorted({
            d.to_string() for d in self._replica_devices
            if self._cluster.get_local_address() == cluster.get_address_from_task(d.job, d.task)
        })
        logging.debug('Local replica devices: {}'.format(self._local_canonical_replica_devices))
        self._num_local_replicas = len(self._local_canonical_replica_devices)

        self._local_worker_id = self._cluster.get_local_worker_task_index()
        self._local_worker_device = '/job:worker/task:{}'.format(self._local_worker_id)

        for synchronizer in self._synchronizers.values():
            synchronizer.assign_cluster_information(self._num_workers, self._num_local_replicas,
                                                    self._local_worker_device, self._local_worker_id,
                                                    is_chief=self._cluster.is_chief())

    def apply(self, graph_item):
        """
        Apply replication to a graph.

        Args:
            graph_item (GraphItem): The graph for replication.

        Returns:
            GraphItem
        """
        new_graph_item = graph_item
        if self._num_local_replicas > 1:
            new_graph_item = self.replicate(graph_item)
            logging.info('Successfully replicated operations')

            # Apply synchronizers
            new_graph_item = self.in_graph_apply(new_graph_item)
            logging.info('Successfully applied local in-graph replication')

        if self._num_workers >= 1:
            new_graph_item = self.between_graph_apply(new_graph_item)
            logging.info('Successfully applied between-graph replication')
        return new_graph_item

    def in_graph_apply(self, graph_item):
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

    def between_graph_apply(self, multi_gpu_graph_item):
        """
        Perform between-graph replication of the graph.

        Args:
            multi_gpu_graph_item (GraphItem): The graph to replication.

        Returns:
            GraphItem
        """
        item = multi_gpu_graph_item.copy()

        with item.graph.as_default():
            with ops.device(self._local_worker_device):
                mirrored_vars = {}
                for gradient, target, update_op in item.var_op_name_to_grad_info.values():
                    syncer_key = strip_replica_prefix(target.name)
                    mirrored_vars[update_op] = self._synchronizers[syncer_key].between_graph_apply(
                        item,
                        update_op,
                        gradient,
                        target
                    )

                resource_variable.gen_mirror_var_init_op(mirrored_vars.values())

                for variable_replicator in mirrored_vars.values():
                    if variable_replicator:
                        variable_replicator.update_colocation_group(item.get_colocation_op)

                self._prune_colocation_groups(item)

        # TODO: make this work
        # update_shard_values_for_worker(num_workers, worker_id)
        return item

    def _replica_device_placer(self, replica_id):
        """A device placer function that places CPU-only ops on CPU instead of destination devices."""
        # strategy device `new_device` merges onto the original `old_device`
        replica_device = self._local_canonical_replica_devices[replica_id]

        def placer(op):
            if all(['CPU' in kernel_def.device_type
                    for kernel_def in kernels.get_registered_kernels_for_op(op.type).kernel]):
                # It assumes an op has a CPU kernel by default.
                new_device = DeviceSpecV2.from_string(self._local_worker_device). \
                    replace(device_type='CPU', device_index=0)
            else:
                new_device = DeviceSpecV2.from_string(replica_device)
            return new_device

        return placer

    def replicate(self, graph_item):
        """
        Replicate the entire graph as many times as num_replica.

        Args:
            graph_item: the original graph item

        Returns: The new graph item
        """
        item = GraphItem(graph=ops.Graph())
        with item.graph.as_default():
            gdef = graph_item.graph.as_graph_def()
            for i in range(self._num_local_replicas):
                with ops.device(self._replica_device_placer(replica_id=i)):
                    import_graph_def(gdef, name=replica_prefix(i))

            # update gradient info
            for i in range(self._num_local_replicas):
                for g_name, t_name in graph_item.grad_target_name_pairs.items():
                    if isinstance(g_name, tuple):
                        new_g_name = (
                            ops.prepend_name_scope(g_name[0], replica_prefix(i)),
                            ops.prepend_name_scope(g_name[1], replica_prefix(i)),
                            ops.prepend_name_scope(g_name[2], replica_prefix(i)))
                    else:
                        new_g_name = ops.prepend_name_scope(g_name, replica_prefix(i))
                    new_t_name = ops.prepend_name_scope(t_name, replica_prefix(i))
                    item.extend_gradient_info_by_names(
                        grads=[new_g_name],
                        targets=[new_t_name]
                    )
                item.info.update(
                    variables=[_from_proto_fn(proto, import_scope=replica_prefix(i)).to_proto()
                               for proto in graph_item.info.variables],
                    replace=False
                )
                item.info.update(
                    table_initializers=[ops.prepend_name_scope(tb_init, replica_prefix(i))
                                        for tb_init in graph_item.info.table_initializers],
                    replace=False
                )
        return item

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
