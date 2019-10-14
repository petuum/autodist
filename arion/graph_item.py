"""GraphItem as metagraph wrapper."""

import contextlib
import functools
from collections import defaultdict
from typing import List

from tensorflow.python.framework import ops
from tensorflow.python.framework.importer import import_graph_def

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common import op_info
from autodist.kernel.common.utils import get_ancestors, get_consumers, parse_name_scope
from autodist.utils import logging


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


# Not a stack structure, thus not supporting nested graph item contexts.
_default_graph_item = None


def get_default_graph_item():
    """Get the default graph item of the current scope."""
    return _default_graph_item


def wrap_gradients(fn):
    """Wrapper for gradients functions in tensorflow.python.ops.gradients_impl."""
    def wrapper(*args, **kwargs):
        if fn.__name__ == 'gradient':
            targets = kwargs.get('sources') or args[2]
        else:
            targets = kwargs.get('xs') or args[1]  # Keep kwargs optional as default
        grads = fn(*args, **kwargs)
        if _default_graph_item:
            _default_graph_item.extend_gradient_info(grads, targets)
            logging.debug('Registered grads: \n {} with targets: \n {}'.format(grads, targets))
        return grads
    return wrapper


class Info:
    """Temp GraphItem Info before RunnerV2."""

    # v1 mode                   # v2 mode
    trainable_variables: List  # variable_captures if trainable
    variables: List  # variable_captures
    table_initializers: List  # Deprecating
    queue_runners: List  # Deprecating

    @property
    def initializers(self):
        """Initializers."""
        return [v.initializer for v in self.variables] + self.table_initializers


class GraphItem:
    """
    GraphItem as TensorFlow Graph wrapper.

    It represents the states in-between consecutive AutoDist kernel graph transformations.
    Graph is the primary property of GraphItem, whereas MetaGraph is exported/generated on demand.
    """

    def __init__(self, graph=None, graph_def=None):
        if graph:
            self._graph = graph
        elif graph_def:
            self._graph = ops.Graph()
            with self._graph.as_default():
                import_graph_def(graph_def, name="")
        else:
            self._graph = ops.Graph()

        # grad tensor name --> variable name  (state-delta tensor name --> stateful op name)
        self._grad_target_pairs = {}

        ###################################
        # Info
        self._info = Info()

    def update_info(self, **kwargs):
        """Set info."""
        replace = kwargs.pop('replace', True)
        if not replace:
            for k, v in kwargs.items():
                getattr(self._info, k).extend(v)
        else:
            self._info.__dict__.update(**kwargs)

    def get_trainable_variables(self):
        """Get variables that need to be synchronized if doing data parallelism."""
        return [op.outputs[0] for op in self.trainable_var_op_to_var]

    @contextlib.contextmanager
    def as_default(self, graph_mode=True):
        """A context scope with current graph item as the default."""
        global _default_graph_item
        if _default_graph_item:
            raise SyntaxError('GraphItem does not support nested contexts.')
        _default_graph_item = self
        # if global graph mode
        if graph_mode:
            with self._graph.as_default():  # enter graph mode
                yield self
        else:
            yield self
        _default_graph_item = None

    def extend_gradient_info(self, grads, targets):
        """Add the detected grad-target pairs to the object."""
        for g, t in zip(grads, targets):
            self._grad_target_pairs[
                (g.indices.name, g.values.name, g.dense_shape.name) if isinstance(g, ops.IndexedSlices) else g.name
            ] = t.name

    def copy_gradient_info_from(self, other):
        """Copy gradient info from the another GraphItem object."""
        # TODO: Future export autodist-defined protobuf message
        self._grad_target_pairs = other._grad_target_pairs.copy()

    @property
    def graph(self):
        """
        Returns the Graph associated with this GraphItem.

        Returns:
            ops.Graph
        """
        return self._graph

    @cached_property
    def all_update_ops(self):
        """
        Get all ops in the graph that perform stateful operations.

        Returns:
            List
        """
        return [op for op in self.graph.get_operations() if
                op.type in op_info.DENSE_VAR_UPDATE_OP_TYPES.keys() | op_info.SPARSE_VAR_UPDATE_OP_TYPES.keys()]

    @cached_property
    def update_op_to_grad_target(self):
        """
        Get all update ops.

        Get all ops in the graph that perform stateful operations,
        on trainable variables, excluding initialization (i.e. the first "update").

        Returns:
            List
        """
        expected_var_ops = {var.op: (grad, var) for grad, var in self.grad_target_pairs}
        update_ops = {}
        for op in self.all_update_ops:
            var_op = op.inputs[op_info.UPDATE_OP_VAR_POS].op
            on_trainable_variable = var_op in expected_var_ops
            var_scope = var_op.name
            update_op_scope = parse_name_scope(op.name)
            is_initialization = update_op_scope == var_scope
            is_saving = update_op_scope.startswith('save')
            if on_trainable_variable and not is_initialization and not is_saving:
                update_ops[op] = expected_var_ops[var_op]
        return update_ops

    @cached_property
    def global_step_update_ops(self):
        """
        Get all ops in the graph that are part of the global step.

        Returns:
            List
        """
        return [
            op for op in self.all_update_ops
            if any((
                'global_step' in input.name or 'iter' in input.name
                for input in op.inputs
            ))
        ]

    @cached_property
    def grad_list(self):
        """
        List of target gradients that will be updated.

        Returns:
            List
        """
        return [grad for grad, _ in self.grad_target_pairs]

    @cached_property
    def target_list(self):
        """
        List of target variables that will be updated.

        Returns:
            List
        """
        return [target for _, target in self.grad_target_pairs]

    @cached_property
    def grad_target_name_pairs(self):
        """
        List of names of grad and target variable pairs.

        Return:
            List
        """
        return self._grad_target_pairs

    @cached_property
    def grad_target_pairs(self):
        """
        List of grad and target variable pairs.

        Return:
             List
        """
        return [(
            ops.IndexedSlices(
                indices=self.graph.get_tensor_by_name(g[0]),
                values=self.graph.get_tensor_by_name(g[1]),
                dense_shape=self.graph.get_tensor_by_name(g[2])
            ) if isinstance(g, tuple) else self.graph.get_tensor_by_name(g),
            self.graph.get_tensor_by_name(t)
        ) for g, t in self._grad_target_pairs.items()]

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
        return {self.graph.get_operation_by_name(var.op.name): var for var in self._info.trainable_variables}

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
        for queue_runner in self.get_ops_in_graph(self._info.queue_runners):
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

    def get_ops_in_graph(self, op_iter):
        """
        Given an iterator of ops or op names, return the corresponding ops in self graph.

        Args:
            op_iter (Iterable): Ops or ops names

        Returns:
            Iterable
        """
        return type(op_iter)((self.graph.get_operation_by_name(o if isinstance(o, str) else o.name) for o in op_iter))
