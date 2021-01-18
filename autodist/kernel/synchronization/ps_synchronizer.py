# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# It includes the derived work based on:
# https://github.com/snuspl/parallax
# Copyright (C) 2018 Seoul National University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PS Synchronizer."""
from functools import partial
from typing import List

from tensorflow.python import ops
from tensorflow.python.framework import device_spec, dtypes, constant_op
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import math_ops, data_flow_ops, gen_control_flow_ops, \
    control_flow_ops, gen_math_ops, gen_array_ops

from autodist.const import MAX_INT64, AUTODIST_PREFIX
from autodist.kernel.common import utils, variable_utils
from autodist.kernel.common.op_info import UPDATE_OP_VAR_POS
from autodist.kernel.common.proxy_variable import ProxyVariable
from autodist.kernel.common.utils import get_op_name, get_consumers, get_ancestors, traverse, update_consumers, \
    update_control_consumers, replica_prefix, strip_replica_prefix, get_control_consumers, \
    remove_from_control_consumers, get_index_from_tensor_name, update_colocation_group
from autodist.kernel.common.variable_utils import get_read_var_ops
from autodist.kernel.synchronization.synchronizer import Synchronizer
from autodist.proto import synchronizers_pb2


class PSSynchronizer(Synchronizer):
    """
    PS Synchronizer.

    Synchronizes gradient updates using a Parameter Server.

    For in-graph synchronization, this just aggregates gradients on a worker's CPU.

    For between-graph synchronization, this aggregates gradients on a pre-defined
    (defined in the Strategy) parameter server.

    To keep this gradient aggregation in sync, the chief gives each worker a token
    for each variable for the workers to mark when their variable update is complete.
    """

    def __init__(self, config: synchronizers_pb2.PSSynchronizer):
        self.target_device = config.reduction_destination if config.reduction_destination else ""
        self._local_replication = config.local_replication
        self._sync = config.sync
        self._staleness = config.staleness

        self._var_op_to_agg_grad = {}
        self._var_op_to_accum_apply_op = {}
        super().__init__()

    def in_graph_apply(self, graph_item, var_name):
        """
        Apply in-graph ps synchronization.

        Args:
            graph_item: the old graph item
            var_name: the variable name w/o replica prefix

        Returns:
            graph_item.GraphItem

        """
        item = graph_item
        var_op_name = get_op_name(var_name)
        master_replica_index = 0

        with item.graph.as_default():
            self._prune_control_dependencies(item, var_op_name, master_replica=master_replica_index)
            self._share_variable(item, var_op_name, master_replica=master_replica_index)
            master_var_name = ops.prepend_name_scope(var_name, replica_prefix(master_replica_index))
            master_var_op_name = get_op_name(master_var_name)
            item.updated = True
            grad, target, update_op = item.var_op_name_to_grad_info_v2[master_var_op_name]
            item.var_queried.append(master_var_op_name)
            agg_grad = self._aggregate_gradients(item, old_update_op=update_op, old_grad=grad, old_target=target)

        # update grad_target_pair and variable info
        for i in range(self.num_replicas):
            var_name_to_remove = ops.prepend_name_scope(var_name, replica_prefix(i))
            item.pop_gradient_info(var_name=var_name_to_remove)
            if i != master_replica_index:
                item.info.pop_variable(var_name=var_name_to_remove)
        item.extend_gradient_info(
            grads=[agg_grad],
            targets=[item.graph.get_tensor_by_name(master_var_name)]
        )
        # TODO(Hao): Prune the graph to use unnecessary nodes
        return item

    def _share_variable(self, graph_item, var_op_name, master_replica=0):
        """
        Share the variable on the replica = `master_replica` (default to 0).

        Update inputs of consumers of the variable on replica > 0 to variable on replica=`master_replica`.

        Args:
            graph_item: the old graph item
            var_op_name: the name of the variable op of the variable to be shared
            master_replica: the index of master replica (default to 0)
        """
        for i in range(0, self.num_replicas):
            if i == master_replica:
                continue
            this_var_op_name = ops.prepend_name_scope(var_op_name, replica_prefix(i))
            this_var_op = graph_item.graph.get_operation_by_name(this_var_op_name)

            # Get all read variable ops to this replica variable
            read_var_ops = get_read_var_ops(this_var_op)

            # Get all consumers of its VarhandleOp,
            # excluding ReadVariableOps and those not in its variable scope
            handle_consumers = set(get_consumers(this_var_op))
            handle_consumers.difference_update(set(read_var_ops))
            handle_consumers.difference_update(
                {con for con in handle_consumers if con.name.startswith(this_var_op_name + '/')})
            # We exclude the `update_op` when updating the consumers on the shared variables.
            # Because i) sharing variable indicates sharing its stateful ops correspondingly
            # (so it is ok to remove stateful ops in none-master replica but we just disconnect it)
            # ii) A variable cannot correspond to more than one update ops for now.
            handle_consumers.difference_update(set(graph_item.all_update_ops))

            # update the consumers of all read variable ops to use the read variable ops of replica=master_replica
            for read_var_op in read_var_ops:
                new_read_var_op_name = ops.prepend_name_scope(ops.strip_name_scope(read_var_op.name, replica_prefix(i)),
                                                              replica_prefix(master_replica))
                new_read_var_op = graph_item.graph.get_operation_by_name(new_read_var_op_name)
                consumers = get_consumers(read_var_op)
                update_consumers(consumers, read_var_op.outputs[0], new_read_var_op.outputs[0])
                update_colocation_group(consumers, read_var_op, new_read_var_op)

            # update the consumers of VarhandleOp to use the handle on replica=master_replica
            new_handle_op_name = ops.prepend_name_scope(ops.strip_name_scope(this_var_op_name, replica_prefix(i)),
                                                        replica_prefix(master_replica))
            new_handle_op = graph_item.graph.get_operation_by_name(new_handle_op_name)
            handle_consumers = list(handle_consumers)
            update_consumers(handle_consumers, this_var_op.outputs[0], new_handle_op.outputs[0])
            update_colocation_group(handle_consumers, this_var_op, new_handle_op)

    def _aggregate_gradients(self, graph_item, old_update_op, old_grad, old_target):
        """
        Apply in-graph synchronization to the grad and target in the graph.

        Args:
            graph_item (graph_item.GraphItem): The graph to put the new ops in.
            old_update_op (Op): The update op corresponding to the grad and target.
            old_grad: The gradient object.
            old_target: The target tensor.

        Returns:
            graph_item.GraphItem
        """
        def ctrl_consumers(op):
            return op._control_outputs  # pylint: disable=protected-access

        # Hierarchical reduction at local node
        reduce_to_device = device_spec.DeviceSpecV2.from_string(self.worker_device). \
            replace(device_type='CPU', device_index=0)

        # TODO(Hao): rethink this -- is this correct?
        graph_item.graph.get_operation_by_name(old_target.op.name)._set_device_from_string(self.target_device)
        graph_item.graph.get_operation_by_name(old_update_op.name)._set_device_from_string(self.target_device)

        # Aggregate old_grad
        if isinstance(old_grad, ops.Tensor):
            # Dense gradient
            consumer_ops = graph_item.get_ops_in_graph(old_grad.consumers())
            ctrl_consumer_ops = graph_item.get_ops_in_graph(ctrl_consumers(old_grad.op))
            agg_grad = self._get_aggregated_dense_grad(graph_item, old_grad.name, reduce_to_device)

            # Make gradients consumers to consume the aggregated gradients.
            self._update_gradient_consumers(graph_item,
                                            consumer_ops,
                                            ctrl_consumer_ops,
                                            old_grad.name,
                                            agg_grad)
        elif isinstance(old_grad, ops.IndexedSlices):
            # Sparse gradient
            indices_c_ops = graph_item.get_ops_in_graph(old_grad.indices.consumers())
            indices_cc_ops = graph_item.get_ops_in_graph(ctrl_consumers(old_grad.indices.op))
            values_c_ops = graph_item.get_ops_in_graph(old_grad.values.consumers())
            values_cc_ops = graph_item.get_ops_in_graph(ctrl_consumers(old_grad.values.op))
            agg_grad = self._get_aggregated_sparse_grad(graph_item, old_target.op, old_grad, reduce_to_device)

            self._update_gradient_consumers(graph_item,
                                            indices_c_ops,
                                            indices_cc_ops,
                                            old_grad.indices.name,
                                            agg_grad.indices)
            self._update_gradient_consumers(graph_item,
                                            values_c_ops,
                                            list(set(values_cc_ops).difference(indices_cc_ops)),
                                            old_grad.values.name,
                                            agg_grad.values)
        else:
            raise RuntimeError("Incorrect old_grad.")
        return agg_grad

    def _prune_control_dependencies(self, graph_item, var_op_name, master_replica=0):
        """
        Prune the control dependencies between the train_op on non-master replica and update op.

        Since the replicator will replicate the entire graph, the update op on non-master replica
        will also be replicated. If the train_op on non-master replica is fetched (which is the case
        in our current feed-fetch remap implementation), it will trigger those update ops and result
        in an unnecessary update over the trainable variables.
        This function prunes the control dependencies between train_op and any variable that bases on
        a PS syncer to avoid this situation.
        """
        for i in range(self.num_replicas):
            if i == master_replica:
                continue
            this_var_op_name = ops.prepend_name_scope(var_op_name, replica_prefix(i))
            _, _, update_op = graph_item.var_op_name_to_grad_info_v2[this_var_op_name]
            source_op = self._get_optimizer_source_op(update_op)
            remove_from_control_consumers(get_control_consumers(source_op), source_op)

    @staticmethod
    def _get_optimizer_source_op(update_op):
        """
        Identify the additional no_op between update_op and train_op if it exists (for certain optimizers).

        Args:
            update_op

        Returns:
            source_op: the no_op if existed, otherwise the update_op itself.
        """
        group_deps = [op for op in get_control_consumers(update_op)
                      if 'Adam' in op.name and 'group_deps' in op.name and op.type == 'NoOp']
        source_op = group_deps[0] if group_deps else update_op
        return source_op

    _BETWEEN_GRAPH_APPLY_SCOPE = 'autodist-between'.lower()

    def between_graph_apply(self, graph_item, var_name):
        """
        Apply between-graph synchronization to the target ops in the graph.

        Args:
            graph_item: The current graph.
            var_name: the variable to be synchronized.

        Returns:
            graph_item.GraphItem: updated graph item.
        """
        if not self._sync:
            return graph_item
        item = graph_item
        # here the variable on replica:0 has been shared, so the original var_name won't work
        var_op_name = ops.prepend_name_scope(get_op_name(var_name), replica_prefix(0))
        item.updated = True
        gradient, target, update_op = item.var_op_name_to_grad_info_v2[var_op_name]
        with item.graph.as_default():
            proxy = self._create_proxy(item, gradient, target) if self._local_replication else None
            if proxy:
                proxy.update_colocation_group(item.get_colocation_op)
            with item.graph.name_scope(self._BETWEEN_GRAPH_APPLY_SCOPE):
                self._var_op_to_agg_grad, self._var_op_to_accum_apply_op = \
                    self._get_accumulation_ops(item, gradient, target,
                                               1 if self._staleness > 0 else self.num_workers)
                self.add_sync_op(item, update_op, proxy)
            item.graph._names_in_use.pop(self._BETWEEN_GRAPH_APPLY_SCOPE)
        return item

    def add_sync_op(self, graph_item, var_update_op, variable_replicator=None):
        """
        Adds additional ops needed for synchronous distributed training into current graph.

        Main purpose of additional ops are:
        1. Initialization
        2. Synchronization
        3. Gradient aggregation

        Args:
            graph_item (graph_item.GraphItem): the graph
            var_update_op: The op
            variable_replicator: The dictionary of master variable op name
                -> list of replicated variables, could be None

        Returns:
            None
        """
        this_worker_cpu = device_spec.DeviceSpecV2.from_string(self.worker_device)
        this_worker_cpu = this_worker_cpu.replace(device_type='CPU', device_index=0)

        var_op = var_update_op.inputs[UPDATE_OP_VAR_POS].op
        is_trainable = var_op in graph_item.trainable_var_op_to_var
        source_op = self._get_optimizer_source_op(var_update_op)
        cc = get_control_consumers(source_op)

        with ops.device(var_op.device):
            if self._staleness == 0:
                queue_ops = self._get_queue_ops(var_update_op, source_op, self.is_chief, is_trainable)
            elif self._staleness > 0:
                queue_ops = self._get_queue_ops_stale(var_update_op, source_op, self.is_chief, is_trainable)
            else:
                raise ValueError("staleness should be greater than or equal to 0.")

            # Only dense trainable variables are replicated locally
            if variable_replicator:
                mirror_variable_update_ops = variable_replicator.get_all_update_ops(
                    queue_ops, worker_device=this_worker_cpu)
                with ops.device(this_worker_cpu):
                    finish_op = control_flow_ops.group(*mirror_variable_update_ops)
            else:
                finish_op = control_flow_ops.group(*queue_ops)

        # Place computation ops of aggregated gradients on PS
        # Note that even though this is doing a graph traversal, it is called in such a way that it
        # only traverses from a gradient aggregator op to a gradient application op (or vice versa) --
        # these corresponding ops should always be adjacent in the graph.
        self._place_post_grad_agg_ops(device_spec.DeviceSpecV2.from_string(self.target_device),
                                      self._var_op_to_agg_grad, {var_op: var_update_op} if is_trainable else {})

        # Replace the control input of train_op to be finish_op
        # Note(Hao): this cc is stale, i.e. cc \subset get_control_consumers(source_op)
        update_control_consumers(cc, source_op, finish_op)

    # pylint: disable=too-many-branches
    def _get_queue_ops(self,
                       var_update_op: ops.Operation,
                       source_op: ops.Operation,
                       is_chief: bool,
                       is_trainable: bool) -> List[ops.Operation]:
        """
        Get queue operations for synchronous parameter update.

        Maintain a list of queues of size 1. The chief machine pushes a token to each queue at the beginning
        of each update. The other workers then dequeue a token from their corresponding queue if their gradient
        is sent to the accumulator. The enqueue and dequeue operations are grouped and have to be completed
        before the model moves on to the next step, thus resulting in synchronous parameter update.

        Args:
            var_update_op: The op

        Returns:
            A list of queue operations.
        """
        var_op = var_update_op.inputs[UPDATE_OP_VAR_POS].op

        var_update_sync_queues = \
            [data_flow_ops.FIFOQueue(1, [dtypes.bool], shapes=[[]],
                                     name='%s_update_sync_queue_%d' % (var_op.name, i),
                                     shared_name='%s_update_sync_queue_%d' % (var_op.name, i))
             for i in range(self.num_workers)]

        queue_ops = []
        if is_chief:
            if is_trainable:
                var_update_deps = [self._var_op_to_accum_apply_op[var_op], source_op]
            else:
                var_update_deps = [var_update_op]
            # Chief enqueues tokens to all other workers after executing variable update
            token = constant_op.constant(False)
            with ops.control_dependencies(var_update_deps):
                for i, q in enumerate(var_update_sync_queues):
                    if i != self.worker_id:
                        queue_ops.append(q.enqueue(token))
                    else:
                        queue_ops.append(gen_control_flow_ops.no_op())
        else:
            # wait for execution of var_update_op
            if is_trainable:
                with ops.control_dependencies([self._var_op_to_accum_apply_op[var_op]]):
                    dequeue = var_update_sync_queues[self.worker_id].dequeue()
            else:
                dequeue = var_update_sync_queues[self.worker_id].dequeue()
            queue_ops.append(dequeue)

        return queue_ops

    # pylint: disable=too-many-branches
    def _get_queue_ops_stale(self,
                             var_update_op: ops.Operation,
                             source_op: ops.Operation,
                             is_chief: bool,
                             is_trainable: bool) -> List[ops.Operation]:
        """
        Get queue operations for staleness synchronous parameter update.

        Maintain a list of queues of size equal to <staleness>. At the beginning of each call of this function
        (either by the chief worker or other workers), it checks whether each queue is not full. If yes, it pushes
        a token to each queue. If not, it does nothing (a no_op).
        Then, for the current worker that calls this function, it dequeues a token from its corresponding queue
        (indexed by its worker id).
        The potential enqueue operations and definite dequeue operation are grouped together, and have to be
        finished before the model moves on to the next step.
        As at each invocation of this function, a row of empty space in the list of queues will be filled. Thus
        <staleness> number of consecutive dequeue operations can be done by a worker without blocking, achieving
        stale synchronous parameter update with maximum <staleness> steps difference.

        Args:
            var_update_op: The op

        Returns:
            A list of queue operations.
        """
        var_op = var_update_op.inputs[UPDATE_OP_VAR_POS].op

        var_update_sync_queues = \
            [data_flow_ops.FIFOQueue(self._staleness, [dtypes.bool], shapes=None,
                                     name='%s_update_sync_queue_%d' % (var_op.name, i),
                                     shared_name='%s_update_sync_queue_%d' % (var_op.name, i))
             for i in range(self.num_workers)]

        # Enqueue one token to every queue if all queues are not full.
        def _enqueue_row_op():
            enqueue_ops = []
            for q in var_update_sync_queues:
                enqueue_ops.append(q.enqueue(False))
            enqueue_a_row_ops = control_flow_ops.group(*enqueue_ops)
            return enqueue_a_row_ops

        def _no_op():
            return gen_control_flow_ops.no_op()

        switch_cond = gen_array_ops.identity(True)
        for q in var_update_sync_queues:
            switch_cond = gen_math_ops.logical_and(switch_cond,
                                                   gen_math_ops.less(q.size(),
                                                                     gen_array_ops.identity(self._staleness)))

        enqueue_a_row_ops = control_flow_ops.cond(switch_cond, _enqueue_row_op, _no_op)

        queue_ops = [enqueue_a_row_ops]

        if is_chief:
            if is_trainable:
                var_update_deps = [self._var_op_to_accum_apply_op[var_op], source_op]
            else:
                var_update_deps = [var_update_op]
            with ops.control_dependencies(var_update_deps):
                dequeue = var_update_sync_queues[self.worker_id].dequeue()
        else:
            # wait for execution of var_update_op
            if is_trainable:
                with ops.control_dependencies([self._var_op_to_accum_apply_op[var_op]]):
                    dequeue = var_update_sync_queues[self.worker_id].dequeue()
            else:
                dequeue = var_update_sync_queues[self.worker_id].dequeue()
        queue_ops.append(dequeue)

        return queue_ops

    def _get_aggregated_dense_grad(self, graph_item, grad_name, reduce_to_device):
        grad_op_name = strip_replica_prefix(get_op_name(grad_name))
        output_idx = get_index_from_tensor_name(grad_name)
        grad_ops = [
            graph_item.graph.get_operation_by_name(ops.prepend_name_scope(grad_op_name, replica_prefix(i)))
            for i in range(self.num_replicas)
        ]

        # Aggregate gradients on `reduce_to_device` (usually CPU)
        with ops.device(reduce_to_device):
            grad_sum_op_name = ops.prepend_name_scope(grad_op_name, u"%sAdd" % AUTODIST_PREFIX)
            grad_sum = math_ops.add_n([grad_op.outputs[output_idx] for grad_op in grad_ops], name=grad_sum_op_name)
            grad_avg_op_name = ops.prepend_name_scope(grad_op_name, u"%sDiv" % AUTODIST_PREFIX)
            grad_avg = math_ops.realdiv(grad_sum, self.num_replicas, name=grad_avg_op_name)
        return grad_avg

    def _get_aggregated_sparse_grad(self, graph_item, var_op, grad, reduce_to_device):
        indices_op_name = strip_replica_prefix(get_op_name(grad.indices.name))
        values_op_name = strip_replica_prefix(get_op_name(grad.values.name))
        dense_shape_op_name = strip_replica_prefix(get_op_name(grad.dense_shape.name))

        indexed_slices_grads = []
        for i in range(self.num_replicas):
            indices_op = graph_item.graph.get_operation_by_name(
                ops.prepend_name_scope(indices_op_name, replica_prefix(i)))
            values_op = graph_item.graph.get_operation_by_name(
                ops.prepend_name_scope(values_op_name, replica_prefix(i)))
            dense_shape_op = graph_item.graph.get_operation_by_name(
                ops.prepend_name_scope(dense_shape_op_name, replica_prefix(i)))
            indexed_slices_grads.append(
                ops.IndexedSlices(
                    values_op.outputs[utils.get_index_from_tensor_name(grad.values.name)],
                    indices_op.outputs[utils.get_index_from_tensor_name(grad.indices.name)],
                    dense_shape_op.outputs[utils.get_index_from_tensor_name(grad.dense_shape.name)])
            )

        return self._aggregate_sparse_gradients(var_op, reduce_to_device, indexed_slices_grads, values_op_name)

    def _aggregate_sparse_gradients(self, var_op, reduce_to_device, indexed_slices_grads, values_op_name):
        with ops.device(reduce_to_device):
            grad_accum_op_name = ops.prepend_name_scope(values_op_name, u"%sAccum" % AUTODIST_PREFIX)
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
                dtype=indexed_slices_grads[0].values.dtype,
                shape=var_op.outputs[0].shape,
                shared_name=grad_accum_op_name,
                name=grad_accum_op_name)
            accum_apply_ops = [grad_accum.apply_indexed_slices_grad(indexed_slices_grads[i],
                                                                    MAX_INT64,
                                                                    name=ops.prepend_name_scope(
                                                                        values_op_name,
                                                                        u"%s-Accum-Apply" % replica_prefix(i)))
                               for i in range(self.num_replicas)]
            take_grad_op_name = ops.prepend_name_scope(values_op_name, u"%sTake-Grad" % AUTODIST_PREFIX)
            with ops.control_dependencies(accum_apply_ops):
                take_grad = grad_accum.take_indexed_slices_grad(self.num_replicas, name=take_grad_op_name)

            new_indices = take_grad.indices
            new_values = take_grad.values
            new_dense_shape = take_grad.dense_shape
            if indexed_slices_grads[0].indices.dtype != new_indices.dtype:
                new_indices = math_ops.cast(
                    new_indices,
                    indexed_slices_grads[0].indices.dtype,
                    name=ops.prepend_name_scope(
                        values_op_name,
                        u"%sTake-Grad-Cast-Indices" % AUTODIST_PREFIX)
                )
            if indexed_slices_grads[0].dense_shape.dtype != new_dense_shape.dtype:
                new_dense_shape = math_ops.cast(
                    new_dense_shape,
                    indexed_slices_grads[0].dense_shape.dtype,
                    name=ops.prepend_name_scope(
                        values_op_name,
                        u"%sTake-Grad-Cast-Shape" % AUTODIST_PREFIX)
                )
        return ops.IndexedSlices(new_values, new_indices, new_dense_shape)

    def _create_proxy(self, graph_item, gradient, target):
        # Do not replicate sparse variables
        if not isinstance(gradient, ops.Tensor) \
                or self.worker_device in self.target_device:  # meaning the variable is local
            return None
        d = device_spec.DeviceSpecV2.from_string(self.worker_device)
        master_var = graph_item.trainable_var_op_to_var.get(target.op)
        master_var_device = device_spec.DeviceSpecV2.from_string(master_var.device)
        device_type = 'GPU' \
            if master_var_device.device_type and master_var_device.device_type.upper() == 'GPU' \
            else 'CPU'
        device_index = 0 if device_type == 'CPU' else master_var_device.device_index
        proxy_var_device = device_spec.DeviceSpecV2(job=d.job,
                                                    replica=d.replica,
                                                    task=d.task,
                                                    device_type=device_type,
                                                    device_index=device_index)
        return ProxyVariable(master_var, graph_item, proxy_var_device)

    @staticmethod
    def _get_accumulation_ops(graph_item, gradient, target, num_accum_required):
        def _get_accum_apply_and_agg_grad(var_op, grad, indices, dense_shape):
            if indices is None:
                tensor = variable_utils.get_read_var_tensor(var_op)
                grad_accum = data_flow_ops.ConditionalAccumulator(
                    grad.dtype,
                    shape=tensor.get_shape(),
                    shared_name=var_op.name + "/grad_accum")
                # Get a copy of consumers list before creating accum_apply_op
                grad_consumers = list(grad.consumers())
                accum_apply_op = grad_accum.apply_grad(
                    grad, local_step=MAX_INT64,
                    name=grad.op.name + '_accum_apply_grad')
                agg_grad = grad_accum.take_grad(num_accum_required,
                                                name=var_op.name + '_take_grad')
                update_consumers(grad_consumers, grad, agg_grad)
                update_control_consumers(get_control_consumers(grad.op),
                                         grad.op, agg_grad.op)
            else:
                grad_indexed_slices = ops.IndexedSlices(values=grad, indices=indices,
                                                        dense_shape=dense_shape)
                grad_accum = data_flow_ops.SparseConditionalAccumulator(
                    grad.dtype,
                    shape=grad.shape,
                    shared_name=var_op.name + "/grad_accum")
                # Get a copy of consumers list before creating accum_apply_op
                indices_consumers = list(indices.consumers())
                grad_consumers = list(grad.consumers())
                accum_apply_op = grad_accum.apply_indexed_slices_grad(
                    grad_indexed_slices, local_step=MAX_INT64,
                    name=grad.op.name + '_accum_apply_grad')
                agg_grad = grad_accum.take_indexed_slices_grad(
                    num_accum_required, name=var_op.name + '_take_grad')
                agg_indices = agg_grad.indices
                if indices.dtype != agg_grad.indices.dtype:
                    agg_indices = math_ops.cast(agg_grad.indices, indices.dtype)
                agg_grad = ops.IndexedSlices(values=agg_grad.values,
                                             indices=agg_indices,
                                             dense_shape=agg_grad.dense_shape)
                assert isinstance(agg_grad, ops.IndexedSlices)
                update_consumers(indices_consumers, indices, agg_grad.indices)
                update_consumers(grad_consumers, grad, agg_grad.values)
                update_control_consumers(get_control_consumers(indices.op),
                                         indices.op, agg_grad.indices.op)
                update_control_consumers(get_control_consumers(grad.op),
                                         grad.op, agg_grad.values.op)
            return accum_apply_op, agg_grad

        # Aggregate gradients from different workers using ConditionalAccumulator.
        # var_op_to_agg_grad and var_op_to_accum_apply_op are updated.
        var_op_to_agg_grad = {}
        var_op_to_accum_apply_op = {}

        if target.op not in graph_item.trainable_var_op_to_var:
            logging.debug(
                "Gradient for non-trainable variable %s is created, "
                "do not insert accumulator for aggregating this gradient"
                % target.op.name)
            return {}, {}

        var_op = target.op
        if isinstance(gradient, ops.Tensor):
            grad = gradient
            indices = None
            dense_shape = None
        else:
            grad = gradient.values
            indices = gradient.indices
            dense_shape = gradient.dense_shape
        with ops.device(var_op.device), ops.name_scope(""):
            accum_apply_op, agg_grad = _get_accum_apply_and_agg_grad(var_op, grad, indices, dense_shape)
        if indices is None:
            var_op_to_agg_grad[var_op] = (None, agg_grad)
        else:
            var_op_to_agg_grad[var_op] = (agg_grad.indices, agg_grad.values)
        var_op_to_accum_apply_op[var_op] = accum_apply_op
        return var_op_to_agg_grad, var_op_to_accum_apply_op

    @staticmethod
    def _place_post_grad_agg_ops(ps_device, var_op_to_agg_grad, var_op_to_apply_grad_op):
        op_to_task = {}
        agg_grad_ops = []
        for var_op, agg_grad in var_op_to_agg_grad.items():
            var_device = device_spec.DeviceSpecV2.from_string(var_op.device)
            if agg_grad[0] is not None:
                agg_grad_ops.append(agg_grad[0].op)
                op_to_task[agg_grad[0].op] = var_device.task
            agg_grad_ops.append(agg_grad[1].op)
            op_to_task[agg_grad[1].op] = var_device.task

        apply_grad_ops = []
        for var_op, apply_grad_op in var_op_to_apply_grad_op.items():
            var_device = device_spec.DeviceSpecV1.from_string(var_op.device)
            apply_grad_ops.append(apply_grad_op)
            # colocate apply_grad and variable
            apply_grad_op._set_device(var_device)
            op_to_task[apply_grad_op] = var_device.task

        # Make sure that the agg_grad_ops and apply_grad_ops are assigned the same task, if possible
        PSGradientTaskAssigner(op_to_task, agg_grad_ops, apply_grad_ops, ps_device).assign()


class PSGradientTaskAssigner:
    """Make sure that all corresponding PS gradient ops are assigned to the same task."""

    SHARED_TASK_ID = -1  # Default value to use when marking a task as shared across devices

    def __init__(self, op_to_task, agg_grad_ops, apply_grad_ops, ps_device):
        self._op_to_task = op_to_task
        self._agg_grad_ops = agg_grad_ops
        self._apply_grad_ops = apply_grad_ops
        self._ps_device = ps_device

        # Note(gyeongin): Need to include control dependency ops in ancestors and
        # descendants or not?
        self._apply_grad_ancestor_ops = get_ancestors(self._apply_grad_ops, self._agg_grad_ops)
        self._agg_grad_descendant_ops = traverse(self._agg_grad_ops, end_ops=self._apply_grad_ops)
        self._ancestors_diff_descendants = self._apply_grad_ancestor_ops.difference(self._agg_grad_descendant_ops)

        logging.debug(f"apply_grad_ancestor_ops: {len(self._apply_grad_ancestor_ops)}")
        logging.debug(f"agg_grad_descendant_ops: {len(self._agg_grad_descendant_ops)}")
        logging.debug(f"ancestors diff descendants: {len(self._ancestors_diff_descendants)}")

    def assign(self):
        """Bi-directionally traverse the graph and assign tasks to ops."""
        # Parent-to-child traversal
        fn = partial(self.__assign_forward, end_ops=self._apply_grad_ops)
        traverse(self._agg_grad_ops, end_ops=self._apply_grad_ops, neighbors_fn=fn)
        # Child-to-parent traversal
        fn = partial(self.__assign_backward, end_ops=self._agg_grad_ops)
        traverse(self._apply_grad_ops, self._agg_grad_ops, neighbors_fn=fn)

    def __assign_forward(self, curr_op, end_ops=None):
        """Get children of and assign a task for `curr_op`. To be used as the `neighbors_fn` for `traverse`."""
        end_ops = end_ops or set()
        if curr_op in self._op_to_task and curr_op not in end_ops:
            return [consumer for consumer in get_consumers(curr_op) if consumer in self._apply_grad_ancestor_ops]

        placement_reference_ops = {input_tensor.op for input_tensor in curr_op.inputs}. \
            difference(self._ancestors_diff_descendants)

        if not all(ref_op in self._op_to_task for ref_op in placement_reference_ops):
            # At least one of `placement_reference_ops` doesn't have a task assigned yet,
            # so re-add `curr_op` to the queue and wait for them to all have tasks
            return [curr_op]

        self.__assign_task(curr_op, placement_reference_ops)

        if curr_op not in end_ops:
            return [consumer for consumer in get_consumers(curr_op) if consumer in self._apply_grad_ancestor_ops]

        return []

    def __assign_backward(self, curr_op, end_ops=None):
        """Get parents of and assign a task for `curr_op`. To be used as the `neighbors_fn` for `traverse`."""
        end_ops = end_ops or set()
        if curr_op in self._op_to_task and curr_op not in end_ops:
            return [input_tensor.op for input_tensor in curr_op.inputs]

        placement_reference_ops = set(get_consumers(curr_op)).intersection(self._apply_grad_ancestor_ops)

        if not all(ref_op in self._op_to_task for ref_op in placement_reference_ops):
            # At least one of `placement_reference_ops` doesn't have a task assigned yet,
            # so re-add `curr_op` to the queue and wait for them to all have tasks
            return [curr_op]

        self.__assign_task(curr_op, placement_reference_ops)

        if curr_op not in end_ops:
            return [input_tensor.op for input_tensor in curr_op.inputs]

        return []

    def __assign_task(self, curr_op, placement_reference_ops):
        """Given an op, assign it a task based on the task assignments of its reference ops."""
        placement_reference_tasks = [self._op_to_task[ref_op] for ref_op in placement_reference_ops]
        unique_tasks = set(placement_reference_tasks)

        if not unique_tasks:
            raise RuntimeError(f"Should have placement reference for operation {curr_op.name}")

        if len(unique_tasks) == 1:
            curr_op_task = unique_tasks.pop()
            self._op_to_task[curr_op] = curr_op_task
        else:
            # priority: assigned placement > shared
            if self.SHARED_TASK_ID in unique_tasks:
                unique_tasks.remove(self.SHARED_TASK_ID)
            if len(unique_tasks) == 1:
                curr_op_task = unique_tasks.pop()
                self._op_to_task[curr_op] = curr_op_task
            else:
                # multiple device placement -> shared
                assert len(unique_tasks) > 1
                curr_op_task = self.SHARED_TASK_ID
                self._op_to_task[curr_op] = self.SHARED_TASK_ID

        logging.debug(f"post_grad_agg_op {curr_op.name} is assigned to ps task {curr_op_task}")
        if curr_op_task == self.SHARED_TASK_ID:
            # TODO: do not assign all shared ops to task 0
            # - we can do better
            curr_op_task = 0

        ps_device = self._ps_device.replace(task=curr_op_task)
        curr_op._set_device(ps_device)
