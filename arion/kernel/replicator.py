"""Replicator."""

from tensorflow.python import ops, import_graph_def
from tensorflow.python.framework import device_spec, kernels
from tensorflow.python.framework.device_spec import DeviceSpecV2
from tensorflow.python.ops.resource_variable_ops import _from_proto_fn

from autodist.graph_item import GraphItem
from autodist.kernel.common.utils import replica_prefix
from autodist.kernel.kernel import Kernel
from autodist.utils import logging


class Replicator(Kernel):
    """Replicator."""

    def __init__(self, key, graph_item, config, cluster):
        super().__init__(key)
        self._graph_item = graph_item
        self._cluster = cluster

        self._replica_devices = {device_spec.DeviceSpecV2.from_string(s) for s in config}

        self._local_canonical_replica_devices = sorted({
            d.to_string() for d in self._replica_devices
            if self._cluster.get_local_address() == cluster.get_address_from_task(d.job, d.task)
        })
        logging.debug('Local replica devices: {}'.format(self._local_canonical_replica_devices))
        self._num_local_replicas = len(self._local_canonical_replica_devices)

        self._local_worker_id = self._cluster.get_local_worker_task_index()
        self._local_worker_device = '/job:worker/task:{}'.format(self._local_worker_id)

    def _apply(self, *args, **kwargs):
        """
        Apply replication to a graph.

        Returns:
            GraphItem
        """
        new_graph_item = self._graph_item
        if self._num_local_replicas >= 1:
            new_graph_item = self.replicate(new_graph_item)
            logging.info('Successfully replicated operations')
        return new_graph_item

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
