"""GraphItem as metagraph wrapper."""

import contextlib
import functools
from collections import defaultdict
import copy

from tensorflow.core.framework.variable_pb2 import VariableDef
from tensorflow.python.framework import ops
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops.variables import Variable

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common import op_info
from autodist.kernel.common.utils import parse_name_scope, get_op_name
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

    def __init__(self):
        # v1 mode                   # v2 mode
        self.trainable_variables = []  # variable_captures if trainable
        self.variables = []  # variable_captures
        self.table_initializers = []  # deprecating

    @property
    def initializers(self):
        """Initializers."""
        return [v.initializer_name for v in self.variables] + self.table_initializers

    def _add_variable(self, var):
        """Add a variable to info tracker."""
        if isinstance(var, VariableDef):
            proto = var
        elif isinstance(var, dict):
            proto = VariableDef()
            for k, v in var.items():
                setattr(proto, k, v)
        else:
            proto = var.to_proto()
        if proto.trainable:
            self.trainable_variables.append(proto)
        self.variables.append(proto)

    def _reset(self):
        self.trainable_variables = []
        self.variables = []
        self.table_initializers = []

    def update(self, variables=None, table_initializers=None, replace=True, **kwargs):
        """Set info."""
        logging.warning('Unused kwargs in info update: {}'.format(kwargs))
        if replace:
            self._reset()
        if variables:
            for v in variables:
                self._add_variable(v)
        if table_initializers:
            for op in table_initializers:
                self.table_initializers.append(op.name if isinstance(op, ops.Operation) else op)

    def copy(self):
        """Copy info."""
        return copy.deepcopy(self)


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
        self.info = Info()

    def copy(self):
        """Get a duplicated current GraphItem."""
        g = GraphItem(graph_def=self._graph.as_graph_def())
        g.info = self.info.copy()
        g._grad_target_pairs = self._grad_target_pairs.copy()
        return g

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

    def extend_gradient_info_by_names(self, grads, targets):
        """Add the detected grad-target pairs to the object by names."""
        for g, t in zip(grads, targets):
            self._grad_target_pairs[g] = t

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
    def var_op_name_to_grad_info(self):
        """A mapping from VarHandleOp name (e.g. "W" not "W:0") to its (grad, var, update_op) tuple."""
        expected_var_ops = {var.op: (grad, var) for grad, var in self.grad_target_pairs.items()}
        res = {}
        for op in self.all_update_ops:
            var_op = op.inputs[op_info.UPDATE_OP_VAR_POS].op
            on_trainable_variable = var_op in expected_var_ops
            var_scope = var_op.name
            update_op_scope = parse_name_scope(op.name)
            is_initialization = update_op_scope == var_scope
            is_saving = update_op_scope.startswith('save')
            if on_trainable_variable and not is_initialization and not is_saving:
                # TODO: Support One Var -> Multiple Grad Update Ops
                res[var_op.name] = expected_var_ops[var_op] + (op, )
        return res

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
        return list(self.grad_target_pairs.keys())

    @cached_property
    def target_list(self):
        """
        List of target variables that will be updated.

        Returns:
            List
        """
        return list(self.grad_target_pairs.values())

    @cached_property
    def grad_target_name_pairs(self):
        """
        List of names of grad and target variable pairs.

        Return:
            List
        """
        return self._grad_target_pairs.copy()

    @cached_property
    def grad_target_pairs(self):
        """
        List of grad and target variable pairs.

        Return:
             List
        """
        return {
            ops.IndexedSlices(
                indices=self.graph.get_tensor_by_name(g[0]),
                values=self.graph.get_tensor_by_name(g[1]),
                dense_shape=self.graph.get_tensor_by_name(g[2])
            ) if isinstance(g, tuple) else self.graph.get_tensor_by_name(g): self.graph.get_tensor_by_name(t)
            for g, t in self._grad_target_pairs.items()}

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
        return {self.graph.get_operation_by_name(get_op_name(var_def.variable_name)): Variable.from_proto(var_def)
                for var_def in self.info.trainable_variables}

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
