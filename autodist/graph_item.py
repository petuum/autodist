"""GraphItem and its supporting functionality."""

import contextlib
import copy
import functools
from typing import Union, Callable

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.variable_pb2 import VariableDef
from tensorflow.python.framework import ops
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops.resource_variable_ops import _from_proto_fn
from tensorflow.python.ops.variables import Variable

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common import op_info
from autodist.kernel.common.utils import parse_name_scope, get_op_name
from autodist.utils import logging

VariableType = Union[VariableDef, dict, Variable]


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


def wrap_optimizer_init(fn: Callable):
    """Wraps the __init__ function of OptimizerV2 objects and stores the info in the default GraphItem."""
    def wrapper(*args, **kwargs):
        # args[0] should be `self`, which is an object of type == optimizer class
        containing_class = type(args[0])
        class_name = containing_class.__name__

        # For calls like super(AdamWeightDecay, self).__init__(*args, **kwargs), the containing_class.__name__
        # returns the current class (AdamWeightDecay) instead of the parent class (Adam).
        # Avoid patching this pattern by checking fn.__qualname__.
        if not fn.__qualname__.startswith(class_name):
            return fn(*args, **kwargs)
        if _default_graph_item and kwargs.pop('update', True):
            _default_graph_item.extend_optimizer_info(containing_class, *args, **kwargs)
            logging.debug('Registered optimizer: {} \nwith args: {} \nkwargs: {}'.format(class_name, args, kwargs))
        return fn(*args, **kwargs)
    return wrapper


def wrap_optimizer_apply_gradient(fn: Callable):
    """Wraps the apply_gradients function of OptimizerV2 objects and stores the info in the default GraphItem."""
    # Signature for apply_gradients
    # apply_gradients(self, grads_and_vars, name=None)
    def wrapper(*args, **kwargs):
        # Assume grads_and_vars is an iterable of tuples
        # Materialize here because in case it's a generator, we need to be able to iterate multiple times
        grads_and_vars = list(kwargs.get('grads_and_vars') or args[1])
        grads, variables = map(list, zip(*grads_and_vars))
        if _default_graph_item and kwargs.pop('update', True):
            _default_graph_item.extend_gradient_info(grads, variables)
            logging.debug('Registered grads: \n {} with targets: \n {}'.format(grads, variables))
        args = (args[0], grads_and_vars)  # Replace possible generator with definite list
        return fn(*args, **kwargs)
    return wrapper


class Info:
    """
    Stores useful variable tracking information.

    In essence, replaces collections, and this way
    we don't have to deal with `MetaGraphs`.
    """

    def __init__(self):
        # v1 mode                   # v2 mode
        self.variables = []  # variable_captures
        self.table_initializers = []  # deprecating

    @property
    def initializers(self):
        """Initializers."""
        return [v.initializer_name for v in self.variables] + self.table_initializers

    @property
    def trainable_variables(self):
        """Trainable Variables."""
        return [v for v in self.variables if v.trainable]

    @property
    def untrainable_variables(self):
        """Untrainable Variables."""
        return [v for v in self.variables if not v.trainable]

    def _add_variable(self, var: VariableType):
        """Add a variable to our info tracker."""
        if isinstance(var, VariableDef):
            proto = var
        elif isinstance(var, dict):
            proto = VariableDef()
            for k, v in var.items():
                setattr(proto, k, v)
        else:
            proto = var.to_proto()
        self.variables.append(proto)

    def pop_variable(self, var_name: str):
        """Pop out a variable by its name from the info tracker."""
        for i, v in enumerate(self.variables):
            if v.variable_name == var_name:
                self.variables.pop(i)
                break

    def update(self, variables=None, table_initializers=None, replace=True, **kwargs):
        """
        Update info.

        Args:
            variables (Iterable[VariableType]): Iterable of variables to insert.
            table_initializers (Iterable(Union[ops.Operation, str]): Initializers for lookup tables.
            replace (bool): Whether or not to overwrite existing contents
            **kwargs: Unused.
        """
        if kwargs:
            logging.warning('Unused kwargs in info update: {}'.format(kwargs))
        if variables:
            if replace:
                self.variables = []
            for v in variables:
                self._add_variable(v)
        if table_initializers:
            if replace:
                self.table_initializers = []
            for op in table_initializers:
                self.table_initializers.append(op.name if isinstance(op, ops.Operation) else op)

    def copy(self):
        """Copy info."""
        return copy.deepcopy(self)


class GraphItem:
    """
    GraphItem is a TensorFlow Graph wrapper.

    It represents the states in-between consecutive AutoDist.kernel graph transformations.
    tf.Graph is the primary property of GraphItem, whereas MetaGraph is exported/generated on demand.

    A GraphItem can be constructed with either a `tf.Graph` or a `GraphDef`.
    """

    def __init__(self, graph: ops.Graph = None, graph_def: GraphDef = None):
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
        self.optimizer, self.optimizer_args, self.optimizer_kwargs = None, None, None

    def copy(self):
        """Get a duplicated current GraphItem."""
        g = GraphItem(graph_def=self._graph.as_graph_def())
        g.info = self.info.copy()
        g.optimizer = self.optimizer
        g.optimizer_args = self.optimizer_args
        g.optimizer_kwargs = self.optimizer_kwargs
        g._grad_target_pairs = self._grad_target_pairs.copy()
        return g

    def get_trainable_variables(self):
        """Get variables that need to be synchronized if doing data parallelism."""
        return [op.outputs[0] for op in self.trainable_var_op_to_var]

    @contextlib.contextmanager
    def as_default(self):
        """A context scope with current graph item as the default."""
        global _default_graph_item
        if _default_graph_item:
            raise SyntaxError('GraphItem does not support nested contexts.')
        _default_graph_item = self
        # if global graph mode
        if isinstance(self._graph, ops.Graph):
            with self._graph.as_default():  # enter graph mode
                yield self
        else:  # case FuncGraph: keep its eager context
            yield self
        _default_graph_item = None

    def extend_optimizer_info(self, optimizer, *args, **kwargs):
        """Add the detected optimizer to the object."""
        self.optimizer = optimizer
        self.optimizer_args = args
        self.optimizer_kwargs = kwargs

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

    def pop_gradient_info(self, var_name: str):
        """Pop out a grad target pair by variable name."""
        for k, v in self._grad_target_pairs.copy().items():
            if v == var_name:
                self._grad_target_pairs.pop(k)

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

    @property
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
            is_saving = update_op_scope.startswith('save')  # is this safe if AutoDist appends "save" in the future?

            # TODO(future): support one variable -> multiple update ops (see AdamWeightDecay optimizer)
            if on_trainable_variable and not is_initialization and not is_saving and not self._is_auxiliary(op):
                if var_op.name in res:
                    raise ValueError('A variable cannot correspond to more than one update op for now.')
                res[var_op.name] = expected_var_ops[var_op] + (op, )
        return res

    def _is_auxiliary(self, update_op: ops.Operation):
        """Check whether a specific update_op is an auxiliary op that should not be considered."""
        # Skip the AssignSub in AdamWeightDecay
        if 'AdamWeightDecay/AdamWeightDecay/' in update_op.name and update_op.type == 'AssignSubVariableOp' and \
                any([control_op in self.all_update_ops for control_op in update_op._control_outputs]):
            return True
        return False

    @property
    def grad_list(self):
        """
        List of target gradients that will be updated.

        Returns:
            List
        """
        return list(self.grad_target_pairs.keys())

    @property
    def target_list(self):
        """
        List of target variables that will be updated.

        Returns:
            List
        """
        return list(self.grad_target_pairs.values())

    @property
    def grad_target_name_pairs(self):
        """
        List of names of grad and target variable pairs.

        Return:
            List
        """
        return self._grad_target_pairs.copy()

    @property
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

    @property
    def trainable_var_op_to_var(self):
        """
        Mapping from trainable variable ops (e.g. VarHandleOps) to the Variables.

        Returns:
            Dict
        """
        with self.graph.as_default():
            return {self.graph.get_operation_by_name(get_op_name(var_def.variable_name)): _from_proto_fn(var_def)
                    for var_def in self.info.trainable_variables}

    def get_colocation_op(self, colocation_group):
        """
        Get the binding op for a given colocation group.

        Args:
            colocation_group (bytes): The colocation group

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
            op_iter (Iterable[Union[ops.Operation, str]]): Ops or ops names

        Returns:
            Iterable
        """
        return type(op_iter)((self.graph.get_operation_by_name(o if isinstance(o, str) else o.name) for o in op_iter))

    def prepare(self):
        """Prepare for building strategy and/or transforming graph."""
        self.info.update(
            variables=self.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES),
            table_initializers=self.graph.get_collection(ops.GraphKeys.TABLE_INITIALIZERS),
            replace=True
        )
