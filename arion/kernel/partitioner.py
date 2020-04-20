"""Variable Partitioner."""
from copy import deepcopy

from google.protobuf.pyext._message import RepeatedScalarContainer
from tensorflow.core.framework import graph_pb2
from tensorflow.python import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs, array_ops, math_ops
from tensorflow.python.util.compat import as_bytes

from autodist.const import AUTODIST_TO_DELETE_SCOPE, COLOCATION_PREFIX
from autodist.graph_item import GraphItem, Info
from autodist.kernel.common.op_info import MUTABLE_STATE_OP_DIRECT_CONSUMER_OPS
from autodist.kernel.common.variable_utils import is_read_var_op
from autodist.kernel.common.utils import get_op_name, get_consumers, update_consumers, parse_name_scope
from autodist.kernel.kernel import Kernel
from autodist.utils import logging


class VariablePartitioner(Kernel):
    """
    Partitions a GraphItem's variables according to the given strategy.

    Essentially, this does a few things:

    1. Reads the Strategy and finds which variables should be partitioned (and how).
    2. Creates new `PartitionedVariables` (essentially a list of `Variables`) and
       replaces the original `Variable` with these new vars. Also splits the original
       gradient to map to each variable shard.
    3. Deletes the original variables and recreates the Optimizer (since Optimizers
       can often have different logic for `PartitionedVariables`, and we don't want
       to rewire the optimizer internals as well).
    4. Returns a new graph and strategy modified to reflect the partitioning.

    Note that this currently does not work with variables that are part of a control
    flow (including batchnorm). In those cases, we currently do not have a way to
    replicate the control flow's `FuncGraph` for our new variables. Thus, we expect
    that any StrategyBuilder does not generate a strategy that tries to partition a
    control flow variable, otherwise this will likely error.
    """

    def __init__(self, key, node_config: RepeatedScalarContainer, graph_item: GraphItem):
        super().__init__(key)
        self.node_config: RepeatedScalarContainer = node_config
        self.graph_item: GraphItem = graph_item
        self.info: Info = graph_item.info.copy()

    def _apply(self, *args, **kwargs):
        """Partition the variables, returning a new GraphItem and a new corresponding Strategy."""
        # Get ops to partition
        vars_to_partition, unpartitioned_vars = self._get_vars_to_partition()

        if not vars_to_partition:
            return self.graph_item, self.node_config

        # Get everything we want to delete
        to_delete = self._get_ops_to_delete(vars_to_partition)

        # In GraphDef, move everything in to_rename under a separate name scope
        # This allows us to create new ops with the to-be-deleted ops' original names
        new_graph_item = self._batch_prepend_name_scope(to_delete, AUTODIST_TO_DELETE_SCOPE)

        # Create new variables and ops in the new graph
        new_graph_item.copy_gradient_info_from(self.graph_item)
        new_vars = self._create_new_vars(new_graph_item, vars_to_partition, unpartitioned_vars)
        # Remove the ops that are marked for deletion
        output_graph_item = self._delete_marked_ops(new_graph_item, AUTODIST_TO_DELETE_SCOPE)

        # Update graph item with proper variable information
        # The new list contains:
        # 1) The new vars we created (`new_vars`)
        # 2) The new vars the optimizer created (`new_globals`)
        # 3) The old untrainable vars that weren't deleted during partitioning (`untrainable_vars`)
        new_vars = set(new_vars)
        new_globals = set(new_graph_item.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
        deleted_tensor_names = {o.outputs[0].name for o in to_delete if o.outputs}
        untrainable_vars = [v for v in self.graph_item.info.untrainable_variables
                            if v.variable_name not in deleted_tensor_names]
        new_var_list = list(new_globals | new_vars) + untrainable_vars

        self.info.update_variables(new_var_list, replace=True)
        output_graph_item.info = self.info.copy()
        output_graph_item.copy_gradient_info_from(new_graph_item)

        logging.info('Successfully partitioned variables')
        return output_graph_item, self.node_config

    def _get_vars_to_partition(self):
        """
        Analyzes the strategy and returns mappings for the vars to partition and the vars to not.

        Returns:
            vars_to_partition (Dict): Mapping of variable names to number of shards for vars to be partitioned.
            unpartitioned_vars (Dict): Mapping from variable name to gradient name of unpartitioned vars.
        """
        vars_to_partition = {}
        unpartitioned_vars = {}
        for node in self.node_config:
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            shards = getattr(synchronizer, 'reduction_destinations', [])
            if len(shards) > 1:
                vars_to_partition[node.var_name] = shards
                logging.info("Partitioning variable {} across {}".format(node.var_name, shards))
            else:
                grad, _, _ = self.graph_item.var_op_name_to_grad_info[get_op_name(node.var_name)]
                unpartitioned_vars[node.var_name] = grad
        return vars_to_partition, unpartitioned_vars

    def _get_ops_to_delete(self, vars_to_partition):
        """
        Get all ops that need to be deleted based on `vars_to_partition`.

        Also keeps the info object up to date.

        Args:
            vars_to_partition (Dict): Mapping of variable names to number of shards for vars to be partitioned.

        Returns:
            Set of ops to be deleted.
        """
        to_delete = set()

        # Mark all ops part of the optimizer for deletion
        update_op_scopes = set()
        opt_name = self.graph_item.optimizer_args[0]._name
        for var_op_name, (_, _, update_op) in self.graph_item.var_op_name_to_grad_info.items():
            top_level_scope_opt = update_op.name[:update_op.name.find(opt_name) + len(opt_name)]
            # An optimizer can create all its relevant ops under the top level optimizer scope
            update_op_scopes.add(top_level_scope_opt)
            #   as well as nested optimizer scopes under each variable name scope
            update_op_scopes.add(var_op_name + '/' + opt_name)

        for var_name in vars_to_partition:
            var_op_name = get_op_name(var_name)
            var_op = self.graph_item.graph.get_operation_by_name(var_op_name)
            var = self.graph_item.trainable_var_op_to_var[var_op]
            update_op = self.graph_item.var_op_name_to_grad_info[var_op_name][2]
            consumers = get_consumers(var_op)

            # Mark var and all its consumers for deletion
            consumers_to_delete = {c for c in consumers if c.type in MUTABLE_STATE_OP_DIRECT_CONSUMER_OPS}
            to_delete.update([var_op, update_op], consumers_to_delete)

            # Update GraphItem Info
            self.info.pop_variable(var.name)

        to_delete.update({o for o in self.graph_item.graph.get_operations()
                          if any(o.name.startswith(top_level_scope) for top_level_scope in update_op_scopes)})
        # If the user uses optimizer.get_gradients, gradients are stored under optimizer_name/gradients.
        # We don't want to delete those.
        # There could be other cases which require this logic to be made more robust, though.
        to_delete = {o for o in to_delete
                     if not any(o.name.startswith(tl_scope + '/gradients/') for tl_scope in update_op_scopes)}
        return to_delete

    def _batch_prepend_name_scope(self, to_rename, new_name_scope):
        """
        Construct a new GraphItem with all ops in `to_rename` under `new_name_scope`.

        Args:
            to_rename (set): Collection of ops to rename
            new_name_scope (str): The new name scope to prepend to all ops

        Returns:
            GraphItem
        """
        og_graph_def = self.graph_item.graph.as_graph_def()
        new_graph_def = graph_pb2.GraphDef()
        new_graph_def.library.Clear()
        new_graph_def.library.CopyFrom(og_graph_def.library)
        control_flow_contexts = {}

        for node in og_graph_def.node:
            op = self.graph_item.graph.get_operation_by_name(node.name)

            # Save control flow context to add it back later
            # Since it is not automatically set based on the attr's in the graph def
            ctx = op._get_control_flow_context()
            if ctx:
                control_flow_contexts[op.name] = ctx

            if op in to_rename:
                node.name = ops.prepend_name_scope(node.name, new_name_scope)

            # Rename inputs
            for idx, input_name in enumerate(node.input):
                input_op = self.graph_item.graph.get_operation_by_name(get_op_name(input_name))
                if input_op in to_rename:
                    node.input[idx] = ops.prepend_name_scope(input_name, new_name_scope)

            # Fix colocation
            for idx, s in enumerate(node.attr['_class'].list.s):
                name = s[len(COLOCATION_PREFIX):].decode('utf-8')
                if self.graph_item.graph.get_operation_by_name(name) in to_rename:
                    node.attr['_class'].list.s[idx] = (COLOCATION_PREFIX +
                                                       as_bytes(ops.prepend_name_scope(name, new_name_scope)))

            new_graph_def.node.append(node)

        # Re-add control flow contexts
        new_graph_item = GraphItem(graph_def=new_graph_def)
        for op in new_graph_item.graph.get_operations():
            if op.name in control_flow_contexts:
                op._set_control_flow_context(control_flow_contexts[op.name])

        return new_graph_item

    # pylint: disable=too-many-locals
    def _create_new_vars(self, new_graph_item, vars_to_partition, unpartitioned_vars):
        """
        Constructs new partitioned variables in `new_graph_item`.

        Fixes each var's corresponding gradient by splitting the gradient.

        Fixes the optimizer by just constructing a new one using the new variables.

        Args:
            new_graph_item (GraphItem): The GraphItem in which to construct the new variables and ops.
            vars_to_partition (Dict): Mapping of variable names to number of shards for vars to be partitioned.
            unpartitioned_vars (Dict): Mapping from variable name to gradient name of unpartitioned vars.

        Returns:
            List of new variables.
        """
        new_grads, new_vars = [], []
        with new_graph_item.graph.as_default():
            for var_name, shards in vars_to_partition.items():
                var_op_name = get_op_name(var_name)
                var_op = self.graph_item.graph.get_operation_by_name(var_op_name)
                var = self.graph_item.trainable_var_op_to_var[var_op]
                gradient = self.graph_item.var_op_name_to_grad_info[var_op_name][0]

                # Create partitioned variable and split gradients
                num_shards = len(shards)
                partitioner = fixed_size_partitioner(num_shards)
                initial_value = new_graph_item.graph.get_tensor_by_name(var.initial_value.name)
                partitioned_var = vs.get_variable(var_op.name, shape=None, initializer=initial_value,
                                                  partitioner=partitioner, validate_shape=False, use_resource=True)
                var_list = partitioned_var._variable_list

                # Distribute the partitioned variable
                for device, var_slice in zip(shards, var_list):
                    var_slice.op._set_device_from_string(device)

                if isinstance(gradient, ops.IndexedSlices):
                    # Sparse variable
                    new_grad = ops.IndexedSlices(
                        indices=new_graph_item.graph.get_tensor_by_name(gradient.indices.name),
                        values=new_graph_item.graph.get_tensor_by_name(gradient.values.name),
                        dense_shape=new_graph_item.graph.get_tensor_by_name(gradient.dense_shape.name)
                    )
                    split_grad = self._split_indexed_slices_v3(new_grad, len(var_list), var.shape[0],
                                                               name=f"gradients/splits/sparse_split_{var_op_name}")
                else:
                    new_grad = new_graph_item.graph.get_tensor_by_name(gradient.name)
                    split_grad = array_ops.split(new_grad, len(var_list),
                                                 name=f"gradients/splits/split_{var_op_name}")

                self._handle_read(new_graph_item, var_op, partitioned_var)

                self._update_node_config(var, var_list)

                self.info.update_variables(var_list, replace=False)
                new_vars.extend(var_list)
                new_grads.extend(split_grad)
                new_graph_item.extend_gradient_info(split_grad, var_list)
                new_graph_item.pop_gradient_info(var.name)
        new_graph_item.info = self.info.copy()
        all_vars, all_grads = new_vars, new_grads
        for var, grad in unpartitioned_vars.items():
            if isinstance(grad, ops.IndexedSlices):
                # Sparse variable
                grad = ops.IndexedSlices(
                    indices=new_graph_item.graph.get_tensor_by_name(grad.indices.name),
                    values=new_graph_item.graph.get_tensor_by_name(grad.values.name),
                    dense_shape=new_graph_item.graph.get_tensor_by_name(grad.dense_shape.name)
                )
            else:
                grad = new_graph_item.graph.get_tensor_by_name(grad.name)
            all_grads.append(grad)
            var = new_graph_item.trainable_var_op_to_var[new_graph_item.graph.get_operation_by_name(get_op_name(var))]
            # TensorFlow expects the following to not mess autodist with the tf.distribute
            if (not hasattr(var, "_distribute_strategy")) or var._distribute_strategy:
                setattr(var, "_distribute_strategy", None)
            all_vars.append(var)
        with new_graph_item.graph.as_default():
            optimizer = self.graph_item.optimizer(*self.graph_item.optimizer_args[1:],
                                                  **self.graph_item.optimizer_kwargs)
            _ = optimizer.apply_gradients(zip(all_grads, all_vars))
        return new_vars

    @staticmethod
    def _handle_read(new_graph_item, var_op, partitioned_var):
        partitioned_var_tensor = partitioned_var.as_tensor()
        for op in get_consumers(var_op):
            op = new_graph_item.graph.get_operation_by_name(
                ops.prepend_name_scope(op.name, AUTODIST_TO_DELETE_SCOPE)
            )
            if op.type == "ResourceGather":
                # Only Resource Variable needs to be taken care of
                #   because ResourceGather consumes resource tensor rather than the tensor of read_var_op
                # Question: Is there any case where the op.type == "ResourceGather"
                #  but we can't use embedding_lookup_v2 to reconstruct the op consuming a partitioned resource
                # The second input to a ResourceGather op is always the indices per the opdef
                emb_lookup = embedding_ops.embedding_lookup_v2(partitioned_var, ids=op.inputs[1])
                update_consumers(get_consumers(op), op.outputs[0], emb_lookup)
            if is_read_var_op(op, version=1):
                # Without our modification, Reference Vars in TF have a read op associated with them.
                # TF can sometimes look for this and expect it to exist (e.g. in graph.as_graph_element)
                # so we add one back to avoid errors.
                # read_out is already the output tensor of the generated identity op
                read_out = array_ops.identity(partitioned_var_tensor,
                                              name=ops.prepend_name_scope("read", var_op.name))
                update_consumers(get_consumers(op), op.outputs[0], read_out)
            elif is_read_var_op(op, version=2):
                read_out = array_ops.identity(partitioned_var_tensor,
                                              name=ops.prepend_name_scope("Read/ReadVariableOp", var_op.name))
                update_consumers(get_consumers(op), op.outputs[0], read_out)

    def _update_node_config(self, var, var_list):
        """
        Updates the strategy to have one synchronizer per variable shard.

        We do this by removing the config corresponding to the old var,
        duplicating it `num_shards` times and changing the `reduction_destinations`
        in each copy accordingly.

        Args:
            var (Variable): the original variable.
            var_list (List[Variable]): the new sharded variables.
        """
        num_shards = len(var_list)
        og_node = next(n for n in self.node_config if n.var_name == var.name)
        conf = [deepcopy(og_node) for _ in range(num_shards)]
        for idx, (v, node) in enumerate(zip(var_list, conf)):
            node.var_name = v.name
            synchronizer = getattr(node, node.WhichOneof('synchronizer'))
            devices = synchronizer.reduction_destinations
            synchronizer.reduction_destinations[:] = [devices[idx]]
        self.node_config.remove(og_node)
        self.node_config.extend(conf)

    def _delete_marked_ops(self, graph_item, name_scope):
        """
        Constructs a new GraphItem with all ops under `name_scope` removed.

        Args:
            graph_item (GraphItem): The current GraphItem.
            name_scope (str): The name scope to remove.

        Returns:
            GraphItem
        """
        graph_def = graph_item.graph.as_graph_def()
        new_graph_def = graph_pb2.GraphDef()
        new_graph_def.library.Clear()
        new_graph_def.library.CopyFrom(graph_def.library)
        control_flow_contexts = {}

        for node in graph_def.node:
            if parse_name_scope(node.name).startswith(name_scope):
                continue

            # Save control flow context to add it back later
            # Since it is not automatically set based on the attr's in the graph def
            op = graph_item.graph.get_operation_by_name(node.name)
            ctx = op._get_control_flow_context()
            if ctx:
                control_flow_contexts[op.name] = ctx

            for idx, input_name in enumerate(node.input):
                if parse_name_scope(input_name).startswith(name_scope):
                    node.input[idx] = ""

            for idx, s in enumerate(node.attr['_class'].list.s):
                name = s[len(COLOCATION_PREFIX):].decode('utf-8')
                if parse_name_scope(name).startswith(name_scope):
                    node.attr['_class'].list.s.remove(s)

            self._prune_graphdef_node_inputs(node)

            new_graph_def.node.append(node)

        # Re-add control flow contexts
        new_graph_item = GraphItem(graph_def=new_graph_def)
        for op in new_graph_item.graph.get_operations():
            if op.name in control_flow_contexts:
                op._set_control_flow_context(control_flow_contexts[op.name])

        return new_graph_item

    @staticmethod
    def _split_indexed_slices(sp_input=None, num_split=None, dim_size=0, name=None):
        size_per_shard = dim_size // num_split
        all_indices = list(range(dim_size))
        indices = [all_indices[0:i * size_per_shard] + all_indices[(i + 1) * size_per_shard:] for i in range(num_split)]
        split_grads = []
        for i in range(0, num_split):
            # `sparse_mask` op might have severe performance issue when the length of the embedding is huge.
            s = array_ops.sparse_mask(sp_input, indices[i], name=name + f"-{i}")
            s._indices = math_ops.floormod(s.indices, size_per_shard, name=name + f"-{i}/mod")
            split_grads.append(s)
        return split_grads

    # @staticmethod
    # def _split_indexed_slices_v2(sp_input=None, num_split=None, dim_size=0, name=None):
    #     ids_per_partition = dim_size // num_split
    #     extras = dim_size % num_split
    #     # Hao: AutoDist PartitionedPS strategy will guarantee even partitions (i.e. extra = 0),
    #     # but for safety we keep the extra for now.
    #     p_assignments = math_ops.maximum(sp_input.indices // (ids_per_partition + 1),
    #                                      (sp_input.indices - extras) // ids_per_partition,
    #                                      name=name)
    #     split_values = data_flow_ops.dynamic_partition(sp_input.values,
    #                                                    p_assignments,
    #                                                    num_split,
    #                                                    name=name + f"-values")
    #     split_indices = data_flow_ops.dynamic_partition(sp_input.indices,
    #                                                     p_assignments,
    #                                                     num_split,
    #                                                     name=name + f"-indices")
    #     split_grads = []
    #     for i in range(0, num_split):
    #         indices = math_ops.floormod(split_indices[i], ids_per_partition, name=name + f"-{i}/indices")
    #         # we must deliberately split the i-th out as a standalone tensor to avoid SparseAccumulator from hanging.
    #         values = array_ops.identity(split_values[i], name=name + f"-{i}/values")
    #         s = ops.IndexedSlices(values, indices, sp_input.dense_shape)
    #         split_grads.append(s)
    #     return split_grads

    @staticmethod
    def _split_indexed_slices_v3(sp_input=None, num_split=None, dim_size=0, name=None):
        ids_per_partition = dim_size // num_split
        extras = dim_size % num_split
        with ops.name_scope(name):
            p_assignments = math_ops.maximum(sp_input.indices // (ids_per_partition + 1),
                                             (sp_input.indices - extras) // ids_per_partition)
            split_grads = []
            for i in range(0, num_split):
                with ops.name_scope(f"part_{i}"):
                    ids_not_in_i = array_ops.where(math_ops.not_equal(p_assignments, i))
                    flat_ids_not_in_i = array_ops.reshape(ids_not_in_i, [-1])
                    flat_ids_not_in_i = math_ops.cast(flat_ids_not_in_i, dtypes.int32)
                    s = array_ops.sparse_mask(sp_input, flat_ids_not_in_i)
                    s._indices = math_ops.floormod(s.indices, ids_per_partition)
                split_grads.append(s)
        return split_grads

    @staticmethod
    def _prune_graphdef_node_inputs(node):
        """
        Remove empty inputs from a NodeDef.

        Args:
            node (NodeDef): a Graphdef node.
        """
        node.input[:] = [s for s in node.input if s]


def fixed_size_partitioner(num_shards, axis=0):
    """
    Partitioner to specify a fixed number of shards along given axis.

    Args:
    num_shards: `int`, number of shards to partition variable.
    axis: `int`, axis to partition on.

    Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope` and `get_variable`.
    """
    def _partitioner(shape, **unused_args):
        partitions_list = [1] * len(shape)
        partitions_list[axis] = min(num_shards, shape.dims[axis].value)
        return partitions_list

    return _partitioner
