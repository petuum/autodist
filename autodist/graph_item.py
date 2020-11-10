# Copyright 2020 Petuum, Inc. All Rights Reserved.
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

"""GraphItem and its supporting functionality."""

import contextlib
import copy
import functools
from collections import defaultdict
from typing import Union, Callable

from google.protobuf.any_pb2 import Any
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.variable_pb2 import VariableDef
from tensorflow.core.protobuf.saver_pb2 import SaverDef
from tensorflow.python.framework import ops
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops.resource_variable_ops import _from_proto_fn
from tensorflow.python.ops.variables import Variable

from autodist.const import COLOCATION_PREFIX
from autodist.kernel.common import op_info
from autodist.kernel.common.utils import parse_name_scope, get_op_name
from autodist.proto import graphitem_pb2
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
_default_graph_item: None = None


def get_default_graph_item():
    """
    Get the current default graph_item under the graph_item scope.

    Returns:
        GraphItem
    """
    return _default_graph_item


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
        self.savers = []  # for saver

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

    def _add_savers(self, saver):
        if isinstance(saver, SaverDef):
            proto = saver
        else:
            proto = saver.to_proto()
        self.savers.append(proto)

    def pop_variable(self, var_name):
        """Pop out a variable by its name from info tracker."""
        for i, v in enumerate(self.variables):
            if v.variable_name == var_name:
                self.variables.pop(i)
                break

    def update_variables(self, variables, replace=True):
        """
        Update variables in GraphItem Info.

        Args:
            variables (Iterable[VariableType]): Iterable of variables to insert.
            replace (bool): Whether or not to overwrite existing contents.
        """
        if replace:
            self.variables = []
        for v in variables:
            self._add_variable(v)

    def update_table_initializers(self, table_initializers, replace=True):
        """
        Update table initializers in GraphItem Info.

        Args:
            table_initializers (Iterable(Union[ops.Operation, str]): Initializers for lookup tables.
            replace (bool): Whether or not to overwrite existing contents.
        """
        if replace:
            self.table_initializers = []
        for op in table_initializers:
            self.table_initializers.append(op.name if isinstance(op, ops.Operation) else op)

    def update_savers(self, savers, replace=True):
        """
        Update savers in GraphItem Info.

        Args:
            savers (Iterable[SaverType]): Iterable of saverdefs to insert.
            replace: Whether or not to overwrite existing contents.
        """
        if replace:
            self.savers = []
        for s in savers:
            self._add_savers(s)

    def copy(self):
        """Copy info."""
        return copy.deepcopy(self)

    def __eq__(self, other):
        return all([
            self.variables == other.variables,
            self.initializers == other.initializers,
            self.savers == other.savers
        ])


class GraphItem:
    """
    GraphItem is a TensorFlow Graph wrapper.

    It represents the states in-between consecutive AutoDist.kernel graph transformations.
    tf.Graph is the primary property of GraphItem, whereas MetaGraph is exported/generated on demand.

    A GraphItem can be constructed with either a `tf.Graph` or a `GraphDef`.
    """

    # pylint: disable=too-many-instance-attributes
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
        # Optimizing the var_op_name_to_grad query.
        # used to inform the var_op_name_to_grad_dict that the graph has been modified
        # only used when the synchronizer is calling the lookup with optimize=True
        self.updated = True
        # used to cached the result of var_op_name_to_grad function from last time
        self.var_op_name_to_grad_dict = dict()
        # map the updated op to its inputs variables, used to optimize var_op_name_to_grad
        self.update_op_depend_var = defaultdict(list)

        # on if this graph is in loop optimize mode for the first time
        self.first_time_loop = True
        self.loop_phase = False
        self.var_queried = []
        self.useful_update_op = []

    def set_optimize(self):
        """Start a loop of synchronizer apply."""
        self.first_time_loop = True
        self.loop_phase = True

    def reset_optimize(self):
        """End a loop of synchronizer apply."""
        self.first_time_loop = True
        self.loop_phase = False

    def get_trainable_variables(self):
        """Get variables that need to be synchronized if doing data parallelism."""
        return [op.outputs[0] for op in self.trainable_var_op_to_var]

    def get_all_variables(self):
        """Get all variables in this graph item."""
        with self.graph.as_default():
            return [_from_proto_fn(var_def) for var_def in self.info.variables]

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
        if (not self.updated and not self.loop_phase):
            return self.var_op_name_to_grad_dict
        expected_var_ops = {var.op: (grad, var) for grad, var in self.grad_target_pairs.items()}
        res = {}
        for op in self.all_update_ops:
            var_op = op.inputs[op_info.UPDATE_OP_VAR_POS].op
            on_trainable_variable = var_op in expected_var_ops
            var_scope = var_op.name
            update_op_scope = parse_name_scope(op.name)
            is_initialization = update_op_scope == var_scope
            # TODO: we should not hardcode this scope.
            # It is actually coming from the name given to the saver
            is_saving = update_op_scope.endswith('save')

            # TODO(future): support one variable -> multiple update ops (see AdamWeightDecay optimizer)
            if on_trainable_variable and not is_initialization and not is_saving and not self._is_auxiliary(op):
                if var_op.name in res:
                    raise ValueError('A variable cannot correspond to more than one update op for now.')
                res[var_op.name] = expected_var_ops[var_op] + (op,)
        self.updated = False
        self.var_op_name_to_grad_dict = res
        return res

    @property
    def var_op_name_to_grad_info_v2(self):
        """
        An optimized version that is aware of this method is iteratively used. It optimize based on.

        (1) Give an updated option, if the graph has not been updated before this query, then it will not
            calculate again. A new method considering this v2 method needs to manipulate updated outside.
        (2) Give an var_queried option, which will record which variable has been doen synchronized. If all 
            the variable associated with an update op has been synchronized, this method will not consier
            the update op next time (it will reconsider if the current loop has done processed, so the
            set/reset optimize method is necessary to set the boolean flags). This optimization is
            inspired by that the for loop in this method is executed for every update_op, which is typically
            a lot, and causes the slowness. This option is safe in that if the var_queried is not set outside,
            it will not trigger the remove op.
        """
        # if the graph has not been rewritten, return old dict instead of generating a new one
        if not self.updated:
            return self.var_op_name_to_grad_dict
        expected_var_ops = {var.op: (grad, var) for grad, var in self.grad_target_pairs.items()}
        res = []
        # keep a list of useful update_op
        if self.first_time_loop:
            self.useful_update_op = self.all_update_ops.copy()
        for op in self.useful_update_op:
            var_op = op.inputs[op_info.UPDATE_OP_VAR_POS].op
            on_trainable_variable = var_op in expected_var_ops
            var_scope = var_op.name
            update_op_scope = parse_name_scope(op.name)
            is_initialization = update_op_scope == var_scope
            is_saving = update_op_scope.endswith('save')
            if on_trainable_variable and not is_initialization and not is_saving and not self._is_auxiliary(op):
                if var_op.name in res:
                    raise ValueError('A variable cannot correspond to more than one update op for now.')
                res.append(var_op.name)
                self.var_op_name_to_grad_dict[var_op.name] = expected_var_ops[var_op] + (op,)
                if self.first_time_loop:
                    self.update_op_depend_var[op].append(var_op.name)
                #analyze what var_ops the op depends on, if all removed, then can remove this op from the loop
                assert len(self.var_queried) <= 1
                if len(self.var_queried) > 0:
                    if var_op.name == self.var_queried[0]:
                        self.var_queried.remove(var_op.name)
                        self.update_op_depend_var[op].remove(var_op.name)
                        if len(self.update_op_depend_var[op]) == 0:
                            self.useful_update_op.remove(op)
        # recalculated the dict, set the indicator
        self.updated = False
        self.first_time_loop = False
        return self.var_op_name_to_grad_dict

    def _is_auxiliary(self, update_op: ops.Operation):
        """Check whether a specific update_op is an auxiliary op that should not be considered."""
        # Skip the AssignSub in AdamWeightDecay
        if 'AdamWeightDecay/AdamWeightDecay/' in update_op.name and update_op.type == 'AssignSubVariableOp' and \
                any([control_op in self.all_update_ops for control_op in update_op._control_outputs]):
            return True
        return False

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
        self.info.update_variables(self.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), replace=True)
        self.info.update_table_initializers(self.graph.get_collection(ops.GraphKeys.TABLE_INITIALIZERS), replace=True)

    def serialize(self, path):
        """Serialize a graph_item to a specific proto string down to a file path."""
        item_def = graphitem_pb2.GraphItem()
        # GraphDef
        item_def.graph_def.Pack(self.graph.as_graph_def())
        # Grad Target Pairs
        for k, v in self._grad_target_pairs.items():
            if isinstance(k, tuple):
                k = ';'.join(k)
            item_def.grad_target_pairs[k] = v

        # Info
        def f(v, repeated_any):
            a = Any()
            a.Pack(v)
            repeated_any.append(a)

        for v in self.info.variables:
            f(v, item_def.info.variables)
        for v in self.info.savers:
            f(v, item_def.info.savers)
        item_def.info.table_initializers.extend(self.info.table_initializers)
        logging.warning('GraphItem currently does not serialize optimizer info, '
                        'while optimizer info is only temporarily used for partitioner.')
        # Serialization
        item_def.SerializeToString()
        with open(path, "wb+") as f:
            f.write(item_def.SerializeToString())

    @classmethod
    def deserialize(cls, path):
        """Deserialize a graph_item serialized proto message from a file path."""
        item_def = graphitem_pb2.GraphItem()
        with open(path, "rb") as f:
            item_def.ParseFromString(f.read())
        # GraphDef
        gdef = GraphDef()
        item_def.graph_def.Unpack(gdef)
        g = cls(graph_def=gdef)
        # Grad Target Pairs
        for k, v in item_def.grad_target_pairs.items():
            k = k.split(';')
            k = k[0] if len(k) == 1 else tuple(k)
            g._grad_target_pairs[k] = v
        # Info
        for a in item_def.info.variables:
            v = VariableDef()
            a.Unpack(v)
            g.info.update_variables([v], replace=False)
        for a in item_def.info.savers:
            v = SaverDef()
            a.Unpack(v)
            g.info.update_savers([v], replace=False)
        g.info.update_table_initializers(item_def.info.table_initializers)
        return g
