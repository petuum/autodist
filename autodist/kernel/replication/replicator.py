"""Replicator."""
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework.attr_value_pb2 import AttrValue as pb2_AttrValue
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import ops
from tensorflow.python.framework import device_spec, kernels
from tensorflow.python.framework.device_spec import DeviceSpecV2
from tensorflow.python.util.compat import as_bytes

from autodist.const import COLOCATION_PREFIX
from autodist.graph_item import GraphItem
from autodist.utils import logging
from autodist.kernel.common import resource_variable
from autodist.kernel.common.op_info import UNSTAGE_OP_TYPES, STAGE_OP_TYPES
from autodist.kernel.common.utils import get_op_name, replica_prefix


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
        new_graph_item = self.in_graph_apply(graph_item)
        logging.info('Successfully applied local in-graph replication')
        new_graph_item = self.between_graph_apply(new_graph_item)
        logging.info('Successfully applied between-graph replication')
        return new_graph_item

    def in_graph_apply(self, graph_item):
        """
        Perform in-graph replication of the graph.

        Args:
            graph_item (GraphItem): The graph to replication.

        Returns:
            GraphItem
        """
        # Sanity check
        assert all([get_op_name(g.name) in graph_item.op_names_to_replicate for g in graph_item.grad_list])
        multi_gpu_graph_def = self.construct_multi_gpu_graph_def(graph_item)
        multi_gpu_meta_graph_def = meta_graph_pb2.MetaGraphDef()
        multi_gpu_meta_graph_def.CopyFrom(graph_item.export_meta_graph())
        multi_gpu_meta_graph_def.graph_def.Clear()
        multi_gpu_meta_graph_def.graph_def.CopyFrom(multi_gpu_graph_def)

        new_graph_item = GraphItem(meta_graph=multi_gpu_meta_graph_def)
        new_graph_item.update_info(**graph_item._info.__dict__)
        with new_graph_item.graph.as_default():
            for update_op, (gradient, target) in graph_item.update_op_to_grad_target.items():
                self._synchronizers[target.name].in_graph_apply(
                    new_graph_item,
                    update_op,
                    gradient,
                    target
                )

        return new_graph_item

    def between_graph_apply(self, multi_gpu_graph_item):
        """
        Perform between-graph replication of the graph.

        Args:
            multi_gpu_graph_item (GraphItem): The graph to replication.

        Returns:
            GraphItem
        """
        item = GraphItem(meta_graph=multi_gpu_graph_item.export_meta_graph())
        item.copy_gradient_info_from(multi_gpu_graph_item)
        item.update_info(**multi_gpu_graph_item._info.__dict__)
        with item.graph.as_default():
            with ops.device(self._local_worker_device):
                mirrored_vars = {}
                for update_op, (gradient, target) in item.update_op_to_grad_target.items():
                    mirrored_vars[update_op] = self._synchronizers[target.name].between_graph_apply(
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

    def construct_multi_gpu_graph_def(self, graph_item):
        """
        Given a single GPU GraphItem, construct the graph for multiple GPUs / replicas.

        Args:
            graph_item: The original, single-GPU GraphItem.

        Returns:
            GraphDef
        """
        single_gpu_graph_def = graph_item.graph.as_graph_def()

        multi_gpu_graph_def = graph_pb2.GraphDef()
        multi_gpu_graph_def.library.Clear()
        multi_gpu_graph_def.library.CopyFrom(single_gpu_graph_def.library)

        for node in single_gpu_graph_def.node:
            if node.name in graph_item.op_names_to_share:
                # Make a single copy of this op since it's being shared
                # Keep its original attrs
                multi_gpu_graph_def.node.append(node)
                new_node = multi_gpu_graph_def.node[-1]
                self._update_copied_node_properties(graph_item, new_node, replica_id=0, shared=True)
            elif node.name in graph_item.op_names_to_replicate:
                # Make a copy of this op for each replica
                for replica_id in range(self._num_local_replicas):
                    multi_gpu_graph_def.node.append(node)
                    new_node = multi_gpu_graph_def.node[-1]
                    self._update_copied_node_properties(graph_item, new_node, replica_id=replica_id, shared=False)
            else:
                raise RuntimeError("Should not reach here")

        return multi_gpu_graph_def

    def _update_copied_node_properties(self, graph_item, new_node, replica_id, shared=False):
        if not shared:
            new_node.name = ops.prepend_name_scope(new_node.name, replica_prefix(replica_id))

            # strategy device `new_device` merges onto the original `old_device`
            old_device = DeviceSpecV2.from_string(new_node.device)
            if all(['CPU' in kernel_def.device_type
                    for kernel_def in kernels.get_registered_kernels_for_op(new_node.op).kernel]):
                # It assumes an op has a CPU kernel by default.
                new_device = DeviceSpecV2.from_string(self._local_worker_device).\
                    replace(device_type='CPU', device_index=0)
            else:
                new_device = DeviceSpecV2.from_string(self._local_canonical_replica_devices[replica_id])
            new_node.device = old_device.make_merged_spec(new_device).to_string()

            if new_node.op in STAGE_OP_TYPES + UNSTAGE_OP_TYPES:
                # Shared name is used to allow batching of elements on the same device
                new_node.attr['shared_name'].s = (
                    ops.prepend_name_scope(
                        new_node.attr['shared_name'].s,
                        replica_prefix(replica_id))).encode('utf-8')
            if 'frame_name' in new_node.attr:
                # Frame name identifies this node's computation frame for the TF Executor
                new_node.attr['frame_name'].s = (
                    ops.prepend_name_scope(
                        new_node.attr['frame_name'].s,
                        replica_prefix(replica_id))).encode('utf-8')
        self._set_inputs_replica_id(graph_item, new_node, replica_id=replica_id)
        self._update_colocation(new_node, graph_item.op_names_to_replicate)

    @staticmethod
    def _set_inputs_replica_id(graph_item, new_node, replica_id):
        for idx, input_name in enumerate(new_node.input):
            if get_op_name(input_name) in graph_item.op_names_to_replicate:
                new_node.input[idx] = ops.prepend_name_scope(input_name, replica_prefix(replica_id))

    @staticmethod
    def _update_colocation(node, op_names_to_replicate, replica_id=None):
        if '_class' not in node.attr:
            return
        class_list = node.attr['_class'].list
        to_delete = []
        for idx, s in enumerate(class_list.s):
            if s.startswith(COLOCATION_PREFIX):
                op_name_to_bind_to = s[len(COLOCATION_PREFIX):].decode('utf-8')
                if op_name_to_bind_to in op_names_to_replicate:
                    # delete colocation constraint if shared op needs to be
                    # colocated with replica op
                    if replica_id is None:
                        to_delete.append(s)
                    else:
                        new_op_name_to_bind_to = ops.prepend_name_scope(op_name_to_bind_to,
                                                                        replica_prefix(replica_id))
                        class_list.s[idx] = COLOCATION_PREFIX + as_bytes(new_op_name_to_bind_to)
        for item in to_delete:
            class_list.s.remove(item)

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
