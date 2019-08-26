"""Replicator."""
from tensorflow.core.framework.attr_value_pb2 import AttrValue as pb2_AttrValue
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import ops
from tensorflow.python.framework import device_spec
from tensorflow.python.training.saver import import_meta_graph

from autodist.graph_item import GraphItem
from autodist.kernel.common import resource_variable
from autodist.kernel.common.utils import get_op_name
from autodist.kernel.device.setter import ReplicaDeviceSetter
from autodist.kernel.experimental.helpers import get_ops_to_replicate, \
    construct_multi_gpu_graph_def, handle_collection_def
from autodist.kernel.synchronization.ps_synchronizer import PSSynchronizer


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
        print('# Local replicas:', self._local_canonical_replica_devices)
        self._num_local_replicas = len(self._local_canonical_replica_devices)

        self._local_worker_id = self._cluster.get_local_worker_task_index()

    def apply(self, graph_item):
        """
        Apply replication to a graph.

        Args:
            graph_item (GraphItem): The graph for replication.

        Returns:
            GraphItem
        """
        return self.between_graph_apply(self.in_graph_apply(graph_item))

    def in_graph_apply(self, graph_item):
        """
        Perform in-graph replication of the graph.

        Args:
            graph_item (GraphItem): The graph to replication.

        Returns:
            GraphItem
        """
        with graph_item.graph.as_default() as graph:
            ops_to_replicate = get_ops_to_replicate(graph_item)
            op_names_to_replicate = {op.name for op in ops_to_replicate}

            # Sanity check
            assert all([get_op_name(g.name) in op_names_to_replicate for g in graph_item.grad_list])

            # By default, share all ops not ancestors of the fetches (e.g. stateful ops)
            # These won't be run by session.run, so they don't matter
            ops_to_share = set(graph.get_operations())
            ops_to_share.difference_update(ops_to_replicate)
            op_names_to_share = {op.name for op in ops_to_share}

        multi_gpu_graph_def = \
            construct_multi_gpu_graph_def(graph_item.meta_graph.graph_def, op_names_to_replicate, op_names_to_share,
                                          num_replicas=self._num_local_replicas,
                                          replica_devices=self._local_canonical_replica_devices)
        multi_gpu_meta_graph_def = meta_graph_pb2.MetaGraphDef()
        multi_gpu_meta_graph_def.CopyFrom(graph_item.meta_graph)
        multi_gpu_meta_graph_def.graph_def.Clear()
        multi_gpu_meta_graph_def.graph_def.CopyFrom(multi_gpu_graph_def)

        handle_collection_def(multi_gpu_meta_graph_def, op_names_to_replicate,
                              num_replicas=self._num_local_replicas)

        new_graph_item = GraphItem(meta_graph=multi_gpu_meta_graph_def)
        with new_graph_item.graph.as_default():
            for gradient, target in graph_item.grad_target_pairs:
                self._synchronizers[target.name].in_graph_apply(
                    graph_item,
                    new_graph_item,
                    gradient,
                    target,
                    self._num_local_replicas,
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
        local_worker_device = '/job:worker/task:{}'.format(self._local_worker_id)

        with ops.Graph().as_default() as graph:
            with ops.device(
                    ReplicaDeviceSetter(
                        worker_device=local_worker_device,
                        synchronizers=self._synchronizers
                    )
            ):
                import_meta_graph(multi_gpu_graph_item.meta_graph)
                item = GraphItem(graph=graph)  # For access to update_ops, grad_list, and target_list

                mirrored_vars = {}
                for update_op, (gradient, target) in item.update_op_to_grad_target.items():
                    # TODO REFACTOR: Clean up this signature?
                    mirrored_vars[update_op] = self._synchronizers[target.name].between_graph_apply(
                        item,
                        gradient,
                        target,
                        local_worker_device,
                        self._num_workers,
                        self._num_local_replicas
                    )

                resource_variable.gen_mirror_var_init_op(mirrored_vars.values())

                # TODO: why we have a separate FOR loops compared with the above one?
                for update_op, (_, target) in item.update_op_to_grad_target.items():
                    self._synchronizers[target.name].add_sync_op(
                        item,
                        update_op,
                        self._local_worker_id,
                        local_worker_device,
                        self._num_workers,
                        variable_replicator=mirrored_vars[update_op],
                    )

                for global_step_op in item.global_step_ops:
                    PSSynchronizer().add_sync_op(
                        item,
                        global_step_op,
                        self._local_worker_id,
                        local_worker_device,
                        self._num_workers,
                        variable_replicator={}
                    )

                for variable_replicator in mirrored_vars.values():
                    variable_replicator.update_colocation_group(item.get_colocation_op)

                self.__prune_colocation_groups(item)

            # TODO: make this work
            # update_shard_values_for_worker(num_workers, worker_id)

        return item

    @staticmethod
    def __prune_colocation_groups(graph_item):
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
