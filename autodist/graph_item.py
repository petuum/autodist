"""GraphItem as metagraph wrapper."""

import functools
from collections import defaultdict

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.python.training.saver import export_meta_graph, import_meta_graph

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common import op_info
from autodist.kernel.common.utils import get_ancestors, get_consumers
from autodist.kernel.common.resource_variable import get_read_var_ops


def cached_property(fn, *args, **kwargs):
    """
    Decorator to make a function a "cached property".

    This means that it is a property whose return value is cached after the
    first time it is called.

    Args:
        fn: The function to be made a cached property
        *args: Any args for the function
        **kwargs: Any kwargs for the function

    Returns:
        function
    """
    return property(functools.lru_cache()(fn, *args, **kwargs))


class GraphItem:
    """
    GraphItem as TensorFlow Graph wrapper.

    It represents the states in-between consecutive AutoDist kernel graph transformations.
    Graph is the primary property of GraphItem, whereas MetaGraph is exported/generated on demand.
    """

    def __init__(self, graph=None, meta_graph=None):
        if graph:
            self._graph = graph
        elif meta_graph:
            self._graph = ops.Graph()
            with self._graph.as_default():
                import_meta_graph(meta_graph)
        else:
            self._graph = ops.Graph()

    def get_variables_to_sync(self):
        """Get variables that need to be synchronized if doing data parallelism."""
        with self.graph.as_default():
            return trainable_variables()

    @property
    def graph(self):
        """
        Returns the Graph associated with this GraphItem.

        Returns:
            ops.Graph
        """
        return self._graph

    def export_meta_graph(self):
        """
        Returns the MetaGraph associated with this GraphItem.

        Returns:
            MetaGraph
        """
        return export_meta_graph(graph=self._graph)

    @cached_property
    def all_update_ops(self):
        """
        Get all ops in the graph that perform stateful operations, including GlobalStep ops.

        Returns:
            List
        """
        return [op for op in self.graph.get_operations() if
                op.type in op_info.DENSE_VAR_UPDATE_OP_TYPES.keys() | op_info.SPARSE_VAR_UPDATE_OP_TYPES.keys()]

    @cached_property
    def update_ops(self):
        """
        Get all ops in the graph that perform stateful operations (updates).

        Returns:
            List
        """
        # Filter out `global_step` and `iter` ops
        return [
            op for op in self.all_update_ops
            if not any((
                'global_step' in input.name or 'iter' in input.name
                for input in op.inputs
            ))
        ]

    @cached_property
    def global_step_ops(self):
        """
        Get all ops in the graph that are part of the global step.

        Returns:
            List
        """
        return set(self.all_update_ops).difference(self.update_ops)

    @cached_property
    def grad_list(self):
        """
        List of target gradients that will be updated.

        Returns:
            List
        """
        grads = []
        for op in self.update_ops:
            if op.type in op_info.DENSE_VAR_UPDATE_OP_TYPES:
                input_indices = op_info.DENSE_VAR_UPDATE_OP_TYPES[op.type]
                grads.append(op.inputs[input_indices[0]])
            else:
                input_indices = op_info.SPARSE_VAR_UPDATE_OP_TYPES[op.type]
                # handle IndexSlices
                indices = op.inputs[input_indices[0]]
                values = op.inputs[input_indices[1]]
                handle = op.inputs[op_info.UPDATE_OP_VAR_POS]
                if handle.dtype is dtypes.resource:
                    dense_shape = resource_variable_ops.variable_shape(handle)
                else:
                    dense_shape = handle.shape
                grads.append(ops.IndexedSlices(values, indices, dense_shape))
        return grads

    @cached_property
    def target_list(self):
        """
        List of target variables that will be updated.

        Returns:
            List
        """
        targets = []
        for op in self.update_ops:
            targets.append(op.inputs[op_info.UPDATE_OP_VAR_POS])
        return targets

    @cached_property
    def grad_target_pairs(self):
        """
        List of grad and target variable pairs.

        Return:
             List
        """
        return list(zip(self.grad_list, self.target_list))

    @cached_property
    def update_op_to_grad_target(self):
        """
        Mapping from update ops to a tuple of the corresponding gradient and target.

        Returns:
            Dict
        """
        return {
            update_op: self.grad_target_pairs[idx]
            for idx, update_op in enumerate(self.update_ops)
        }

    @cached_property
    def trainable_vars(self):
        """
        Mapping from ops to the corresponding trainable variables.

        Returns:
            Dict
        """
        with self.graph.as_default():
            trainable_vars = {var.op: var for var in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)}
        return trainable_vars

    @cached_property
    def global_vars(self):
        """
        Mapping from ops to the corresponding global variables.

        Returns:
            Dict
        """
        with self.graph.as_default():
            global_vars = {var.op: var for var in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)}
        return global_vars

    @cached_property
    def control_consumers(self):
        """
        Mapping from an op to the ops for which it is a control dependency.

        A: [B]
        A --> B
        B depends on A

        Args:
            graph: TensorFlow Graph object

        Returns:
            DefaultDict
        """
        op_to_control_consumer_ops = defaultdict(list)
        for op in self.graph.get_operations():
            for control_input_op in op.control_inputs:
                op_to_control_consumer_ops[control_input_op].append(op)
        return op_to_control_consumer_ops

    @cached_property
    def trainable_var_op_to_var(self):
        """
        Mapping from trainable variable ops (e.g. VarHandleOps) to the Variables.

        Returns:
            Dict
        """
        with self.graph.as_default():
            output = {var.op: var for var in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)}
        return output

    @cached_property
    def global_var_op_to_var(self):
        """
        Mapping from global variable ops (e.g. VarHandleOps) to the Variables.

        Returns:
            Dict
        """
        with self.graph.as_default():
            output = {var.op: var for var in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)}
        return output

    @cached_property
    def ops_to_replicate(self):
        """Get ops to be replicated."""
        grad_related = set()
        for grad in self.grad_list:
            if isinstance(grad, ops.IndexedSlices):
                grad_related.add(grad.indices)
                grad_related.add(grad.values)
                grad_related.add(grad.dense_shape)
            elif isinstance(grad, ops.Tensor):
                grad_related.add(grad)
            else:
                raise RuntimeError("Incorrect grad.")

        grads_ancestor_ops = get_ancestors([grad.op for grad in grad_related],
                                           include_control_inputs=True)

        pipeline_ops = self.pipeline_ops(grads_ancestor_ops)

        global_var_related_ops = set()
        for global_var in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES):
            global_var_related_ops.add(global_var.op)
            global_var_related_ops.add(global_var.initializer)
            if global_var.op.type == 'VarHandleOp':
                # TF 2.x
                read_variable_ops = get_read_var_ops(global_var.op)
                global_var_related_ops.update(read_variable_ops)
            else:
                # TF 1.x
                global_var_related_ops.add(global_var._snapshot.op)  # pylint: disable=protected-access

        table_related_ops = set()
        for table_init in ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS):
            table_related_ops.add(table_init)
            table_related_ops.add(table_init.inputs[0].op)

        # Assume that all variables are member of either GLOBAL_VARIABLES
        # or LOCAL_VARIABLES.
        local_var_op_to_var = {var.op: var for var in ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)}
        local_var_ops = set(local_var_op_to_var.keys())
        local_var_ops.intersection_update(grads_ancestor_ops)

        ops_to_replicate = grads_ancestor_ops.copy()
        ops_to_replicate.update(pipeline_ops)

        # var_handles1 = [op for op in ops_to_replicate if op.type == "VarHandleOp"]
        # var_handles2 = [op for op in global_var_related_ops if op.type == "VarHandleOp"]

        ops_to_replicate.difference_update(global_var_related_ops)
        ops_to_replicate.difference_update(table_related_ops)
        ops_to_replicate.update([local_var_op_to_var[var_op].initializer for var_op in local_var_ops])

        return ops_to_replicate

    @cached_property
    def op_names_to_replicate(self):
        """Get the names of ops to be replicated."""
        return {op.name for op in self.ops_to_replicate}

    @cached_property
    def op_names_to_share(self):
        """Get the names of ops to be shared."""
        # By default, share all ops not ancestors of the fetches (e.g. stateful ops)
        # These won't be run by session.run, so they don't matter
        return {op.name for op in set(self.graph.get_operations()).difference(self.ops_to_replicate)}

    # pylint: disable=too-many-locals
    def pipeline_ops(self, in_ops):  # noqa: MC0001
        """[summary]."""
        unstage_dequeue_iterator_queue = [
            op for op in in_ops
            if op.type in op_info.UNSTAGE_OP_TYPES
            or op.type in op_info.DEQUEUE_OP_TYPES
            or op.type in op_info.ITERATOR_OP_TYPES
        ]

        stage_enqueue_iterator_ops_queue = []
        pipeline_ops = set()
        visited = set()

        def _get_shared_name_to_stage_ops(input_ops):
            stage_ops = [op for op in input_ops if op.type in op_info.STAGE_OP_TYPES]
            shared_name_to_stage_ops = {}
            for stage_op in stage_ops:
                shared_name = stage_op.get_attr("shared_name")
                if shared_name not in shared_name_to_stage_ops:
                    shared_name_to_stage_ops[shared_name] = []
                shared_name_to_stage_ops[shared_name].append(stage_op)
            return shared_name_to_stage_ops

        shared_name_to_stage_ops = _get_shared_name_to_stage_ops(self._graph.get_operations())
        queue_name_to_queue_runner = {}
        for queue_runner in ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
            queue_name_to_queue_runner[queue_runner.name] = queue_runner

        while unstage_dequeue_iterator_queue or stage_enqueue_iterator_ops_queue:
            if unstage_dequeue_iterator_queue:
                curr_op = unstage_dequeue_iterator_queue.pop()
                if curr_op in visited:
                    continue
                visited.add(curr_op)

                if curr_op.type in op_info.UNSTAGE_OP_TYPES:
                    stage_shared_name = curr_op.get_attr("shared_name")
                    stage_ops = shared_name_to_stage_ops[stage_shared_name]
                    for stage_op in stage_ops:
                        pipeline_ops.add(stage_op)
                        stage_enqueue_iterator_ops_queue.append(stage_op)
                    # Handle colocation groups of unstage op (NoOp)
                    assert len(curr_op.colocation_groups()) == 1
                    stage_no_op_name = curr_op.colocation_groups()[0][len(COLOCATION_PREFIX):]
                    pipeline_ops.add(self._graph.get_operation_by_name(stage_no_op_name))
                elif curr_op.type in op_info.DEQUEUE_OP_TYPES:
                    queue_ops = [input.op for input in curr_op.inputs if input.op.type in op_info.QUEUE_OP_TYPES]
                    assert len(queue_ops) == 1
                    queue_op = queue_ops[0]
                    queue_runner = queue_name_to_queue_runner[queue_op.name]
                    for enqueue_op in queue_runner.enqueue_ops:
                        pipeline_ops.add(enqueue_op)
                        stage_enqueue_iterator_ops_queue.append(enqueue_op)
                    pipeline_ops.add(queue_runner.close_op)
                    pipeline_ops.add(queue_runner.cancel_op)
                elif curr_op.type in op_info.ITERATOR_OP_TYPES:
                    consumer_ops = get_consumers(curr_op)
                    stage_enqueue_iterator_ops_queue.extend(consumer_ops)
                else:
                    raise RuntimeError("Should not reach here")

            elif stage_enqueue_iterator_ops_queue:
                curr_op = stage_enqueue_iterator_ops_queue.pop()
                if curr_op in visited:
                    continue
                visited.add(curr_op)
                ancestor_ops = get_ancestors([curr_op],
                                             include_control_inputs=True)
                for ancestor_op in ancestor_ops:
                    pipeline_ops.add(ancestor_op)
                    if ancestor_op.type in op_info.UNSTAGE_OP_TYPES \
                            + op_info.DEQUEUE_OP_TYPES + op_info.ITERATOR_OP_TYPES:
                        unstage_dequeue_iterator_queue.append(ancestor_op)
        return pipeline_ops

    def get_colocation_op(self, colocation_group):
        """
        Get the binding op for a given colocation group.

        Args:
            graph_item: The current graph
            colocation_group: The colocation group

        Returns:
            Op
        """
        assert colocation_group.startswith(COLOCATION_PREFIX)
        binding_op_name = colocation_group[len(COLOCATION_PREFIX):].decode('utf-8')
        return self.graph.get_operation_by_name(binding_op_name)

    def get_ops_in_graph(self, op_list):
        """
        Given a list of ops, return the corresponding Ops in this graph.

        Args:
            op_list (list): Ops

        Returns:
            List
        """
        return [self.graph.get_operation_by_name(op.name) for op in op_list]
