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

"""Variable Partitioner."""
import collections
import re

from google.protobuf.pyext._message import RepeatedScalarContainer
from tensorflow.core.framework import graph_pb2
from tensorflow.python import ops
from tensorflow.python.framework import dtypes, versions
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs, array_ops, math_ops
from tensorflow.python.ops.variables import Variable, PartitionedVariable
from tensorflow.python.util.compat import as_bytes

from autodist.const import AUTODIST_TO_DELETE_SCOPE, COLOCATION_PREFIX
from autodist.checkpoint.saver import Saver
from autodist.graph_item import GraphItem, Info
from autodist.kernel.common.op_info import MUTABLE_STATE_OP_DIRECT_CONSUMER_OPS
from autodist.kernel.common.utils import get_op_name, get_consumers, update_consumers, parse_name_scope
from autodist.kernel.common.variable_utils import is_read_var_op
from autodist.kernel.kernel import Kernel
from autodist.utils import logging


class PartitionerConfig():
    """Helper class to conveniently convert between partition list and partition string."""

    def __init__(self, partition_list=None, partition_str=None):
        if partition_list:
            self._partition_list = partition_list
            self._partition_str = self.serialize(partition_list)
        elif partition_str:
            self._partition_list = self.deserialize(partition_str)
            self._partition_str = partition_str
        else:
            raise ValueError('At least and only one of partition_list and partition_str needs to be provided.')

    @staticmethod
    def _check_partition_list(partition_list):
        if not partition_list:
            logging.warning('Partition list is empty.')
            return False
        all_one = True
        active_axis = 0
        for p in partition_list:
            if p == 0:
                return False
            if p > 1:
                all_one = False
                active_axis += 1
        if all_one:
            logging.warning('Partition list is trivial -- num_split is 1 on every axis.')
            return False
        if active_axis > 1:
            logging.warning('Currently AutoDist only support partitioning along one axis.')
            return False
        return True

    def serialize(self, partition_list):
        """
        Serialize a partition list to a partition str if it is valid.

        Args:
            partition_list (List): A list of integers indicating how many shards to split along each dimension.

        Returns:
            partition_str (str): A serialized string format of the partition list.
        """
        if self._check_partition_list(partition_list):
            return ','.join(str(x) for x in partition_list)
        else:
            raise ValueError()

    def deserialize(self, partition_str):
        """
        Deserialize a partition string to a partition list and check if it is valid.

        Args:
            partition_str (str): A serialized string format of the partition list.

        Returns:
            partition_list (List): A valid partition list.
        """
        if len(partition_str) == 0:
            raise ValueError('Empty partition string.')
        partition_list = [int(num_split) for num_split in partition_str.split(',')]
        if self._check_partition_list(partition_list):
            return partition_list
        else:
            raise ValueError('Invalid partition list.')

    @property
    def partition_str(self):
        """
        The partition string indicating the partition config.

        Returns:
            str
        """
        return self._partition_str

    @property
    def partition_list(self):
        """
        The partition list indicating the partition config.

        Returns:
            List(int)
        """
        return self._partition_list

    @property
    def num_shards(self):
        """
        The number of total partitions.

        Returns:
            int
        """
        shard = 1
        for i in self.partition_list:
            shard = shard * i
        return shard

    @property
    def axis(self):
        """
        The axis to partition along with.

        Returns:
            int
        """
        idx = 0
        for idx, p in enumerate(self.partition_list):
            if p > 1:
                break
        return idx


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

    def _apply(self, *args, **kwargs):      # pylint: disable-msg=too-many-locals
        """Partition the variables, returning a new GraphItem and a new corresponding Strategy."""
        # Get ops to partition
        vars_to_partition, unpartitioned_vars = self._get_vars_to_partition()

        if not vars_to_partition:
            return self.graph_item, self.node_config

        # Get everything we want to delete
        to_delete, top_update_op_scopes = self._get_ops_to_delete(vars_to_partition)

        # In GraphDef, move everything in to_rename under a separate name scope
        # This allows us to create new ops with the to-be-deleted ops' original names
        new_graph_item = self._batch_prepend_name_scope(to_delete, AUTODIST_TO_DELETE_SCOPE)

        # Create new variables and ops in the new graph
        new_graph_item.copy_gradient_info_from(self.graph_item)
        new_vars, partition_config = self._create_new_vars(new_graph_item, vars_to_partition, unpartitioned_vars)
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

        with self.graph_item.graph.as_default():
            # this can be used to get the shape for partitioned vars
            ori_vars = self.graph_item.get_all_variables()
        with output_graph_item.graph.as_default():
            self._update_save_ops(
                graph_item=output_graph_item,
                ori_vars=ori_vars,
                update_op_scopes=top_update_op_scopes,
                partition_config=partition_config
            )
        logging.info('Successfully partitioned variables')
        return output_graph_item, self.node_config

    @staticmethod
    def _group_partitioned_vars(vars_to_group):
        """Group the ops within the same partition."""
        op_names = [v.op.name for v in vars_to_group]
        partition_pattern = r"part_?\d+"
        group = collections.defaultdict(list)
        for name in op_names:
            name_list = name.split("/")
            # the ops in the optimizer
            if len(name_list) >= 2 and re.match(partition_pattern, name_list[-2]):
                name_list.pop(-2)
                ori_name = "/".join(name_list)
                group[ori_name].append(name)
            # the ops outside the optimizer
            elif re.match(partition_pattern, name_list[-1]):
                name_list.pop(-1)
                ori_name = "/".join(name_list)
                group[ori_name].append(name)
        return group

    @staticmethod
    def _get_paritioned_var_info(ori_vars,      # pylint: disable-msg=too-many-locals
                                 new_vars,
                                 var_group,
                                 update_op_scopes,
                                 partition_config):
        """Get the construction info of all PartitionedVariables."""
        partitioned_vars = dict()
        ori_var_ops_to_vars = {v.op.name: v for v in ori_vars}
        new_var_ops_to_vars = {v.op.name: v for v in new_vars}

        major_version = versions.VERSION.split('.')[0]
        for var_op_name, split_var_names in var_group.items():
            ori_var = ori_var_ops_to_vars[var_op_name]
            # get partition config
            partition_name = ori_var.op.name
            if major_version == "1":
                for prefix in update_op_scopes:
                    split_partition_name = partition_name.split("/")
                    if split_partition_name[-1].startswith(prefix):
                        partition_name = "/".join((split_partition_name[:-1]))
                        break
            elif major_version == "2":
                for prefix in update_op_scopes:
                    if partition_name.startswith(prefix):
                        partition_name = partition_name[len(prefix) + 1:]
                        partition_name = partition_name.split("/")
                        partition_name = "/".join((partition_name[:-1]))
                        break
            else:
                raise ValueError("Unknow version of tensorflow!!")
            pc = partition_config[partition_name]

            # create partitioned_var_info
            partitioned_vars[ori_var.name] = {
                "name": ori_var.op.name,
                "shape": ori_var.shape.as_list(),
                "dtype": ori_var.dtype,
                "var_list": [new_var_ops_to_vars[var_op_name] for var_op_name in split_var_names],
                "partitions": pc._partition_list
            }
            # NOTE: here is a strong assumption: partition vars offset in optimizer follows the naming order!!!
            v_list = partitioned_vars[ori_var.name]["var_list"]
            if not all(v._get_save_slice_info() is not None for v in v_list):
                # set SaveSliceInfo
                v_list.sort(key=lambda x: x.name)
                slice_dim, num_slices = vs._get_slice_dim_and_num_slices(pc._partition_list)
                for i, (var_offset, var_shape) in enumerate(
                        vs._iter_slices(ori_var.shape.as_list(), num_slices, slice_dim)):
                    v = v_list[i]
                    v._set_save_slice_info(
                        Variable.SaveSliceInfo(
                            ori_var.name,
                            ori_var.shape.as_list(),
                            var_offset,
                            var_shape
                        )
                    )
        return partitioned_vars

    def _update_save_ops(       # pylint: disable-msg=too-many-locals
            self,
            graph_item,
            ori_vars,
            update_op_scopes,
            partition_config):

        if not self.info.savers:
            return
        all_vars = graph_item.get_all_variables()
        partitioned_var_group = self._group_partitioned_vars(all_vars)
        partitioned_vars = self._get_paritioned_var_info(
            ori_vars,
            all_vars,
            partitioned_var_group,
            update_op_scopes,
            partition_config
        )
        # new PartitionedVariables
        new_vars = list()
        for _, pv_config in partitioned_vars.items():
            p_var = PartitionedVariable(
                name=pv_config["name"],     # NOTE: this should be the op name
                shape=pv_config["shape"],
                dtype=pv_config["dtype"],
                variable_list=pv_config["var_list"],
                partitions=pv_config["partitions"]
            )
            all_vars = [v for v in all_vars if v not in pv_config["var_list"]]
            new_vars.append(p_var)

        for proto in self.info.savers:
            saver = Saver.from_proto(proto)
            saver._is_built = False
            saver.saver_def = None
            saver._var_list = all_vars + new_vars
            saver.build()

    def _get_vars_to_partition(self):
        """
        Analyzes the strategy and returns mappings for the vars to partition and the vars to not.

        Returns:
            vars_to_partition (Dict): Mapping of variable names to the tuple of partition_str and reduction devices.
            unpartitioned_vars (Dict): Mapping from variable name to gradient name of unpartitioned vars.
        """
        vars_to_partition = {}
        unpartitioned_vars = {}
        for node in self.node_config:
            partitioner = getattr(node, 'partitioner')
            if partitioner:
                reduction_destinations = []
                for part in node.part_config:
                    synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                    if hasattr(synchronizer, 'reduction_destination'):
                        reduction_destinations.append(synchronizer.reduction_destination)
                    else:
                        reduction_destinations.append('')
                vars_to_partition[node.var_name] = (partitioner, reduction_destinations)
                logging.info("Partitioning variable {} with configuration {}".format(node.var_name, partitioner))
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
        top_update_op_scopes = set()
        opt_name = self.graph_item.optimizer_args[0]._name
        for var_op_name, (_, _, update_op) in self.graph_item.var_op_name_to_grad_info.items():
            top_level_scope_opt = update_op.name[:update_op.name.find(opt_name) + len(opt_name)]
            # An optimizer can create all its relevant ops under the top level optimizer scope
            update_op_scopes.add(top_level_scope_opt)
            top_update_op_scopes.add(top_level_scope_opt)
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
        # NOTE: Here we assume the name_scope in saver is the default one.
        to_delete.update({o for o in self.graph_item.graph.get_operations()
                          if o.name.startswith("save/")})
        # If the user uses optimizer.get_gradients, gradients are stored under optimizer_name/gradients.
        # We don't want to delete those.
        # There could be other cases which require this logic to be made more robust, though.
        to_delete = {o for o in to_delete
                     if not any(o.name.startswith(tl_scope + '/gradients/') for tl_scope in update_op_scopes)}
        return to_delete, top_update_op_scopes

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
        partition_config = {}
        with new_graph_item.graph.as_default():
            for var_name, (partition_str, reduction_destinations) in vars_to_partition.items():
                var_op_name = get_op_name(var_name)
                var_op = self.graph_item.graph.get_operation_by_name(var_op_name)
                var = self.graph_item.trainable_var_op_to_var[var_op]
                gradient = self.graph_item.var_op_name_to_grad_info[var_op_name][0]

                # Create partitioned variable and split gradients
                pc = PartitionerConfig(partition_str=partition_str)
                partition_config[var_op_name] = pc

                # Now check compatibility
                if isinstance(gradient, ops.IndexedSlices) and pc.axis != 0:
                    raise ValueError('Embedding variables can only be partitioned along the first axis due to '
                                     'the limitation on the `embedding_lookup_v2` op.')

                initial_value = new_graph_item.graph.get_tensor_by_name(var.initial_value.name)
                # NOTE: to enable the saver, for now we only support partition on the one dimension
                # https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/variables.py#L2915
                partitioned_var = vs.get_variable(var_op.name, shape=None, initializer=initial_value,
                                                  partitioner=lambda pconf=pc, **unused_kwargs: pconf.partition_list,
                                                  validate_shape=False, use_resource=True)
                var_list = partitioned_var._variable_list

                # Distribute the partitioned variable if they have a PS synchornizer
                # Actually maybe this is not necessary
                for var_slice, device in zip(var_list, reduction_destinations):
                    if device:
                        var_slice.op._set_device_from_string(device)

                if isinstance(gradient, ops.IndexedSlices):
                    # Sparse variable
                    new_grad = ops.IndexedSlices(
                        indices=new_graph_item.graph.get_tensor_by_name(gradient.indices.name),
                        values=new_graph_item.graph.get_tensor_by_name(gradient.values.name),
                        dense_shape=new_graph_item.graph.get_tensor_by_name(gradient.dense_shape.name)
                    )
                    split_grad = self._split_indexed_slices_v2(new_grad, len(var_list), var.shape[0],
                                                               name=f"gradients/splits/sparse_split_{var_op_name}")
                else:
                    new_grad = new_graph_item.graph.get_tensor_by_name(gradient.name)

                    # sometimes new_grad will have polymorphic shape (None), so we use the shape of the original var
                    split_grad = self._split_tensor_v2(new_grad, pc.num_shards, var.shape, pc.axis,
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
        return new_vars, partition_config

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
        """Updates the strategy to have the correct name for partitioned vars."""
        og_node = next(n for n in self.node_config if n.var_name == var.name)
        for v, node in zip(var_list, og_node.part_config):
            if node.var_name != v.name:
                node.var_name = v.name

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
    def _split_indexed_slices_v2(sp_input=None, num_split=None, dim_size=0, name=None):
        ids_per_partition = dim_size // num_split
        extras = dim_size % num_split
        with ops.name_scope(name):
            # When the partitioned dim cannot be divided by num_split, the reminders are
            # evenly assigned from the first partition to the last.
            p_assignments = math_ops.maximum(sp_input.indices // (ids_per_partition + 1),
                                             (sp_input.indices - extras) // ids_per_partition)
            split_grads = []
            for i in range(0, num_split):
                with ops.name_scope(f"part_{i}"):
                    ids_not_in_i = array_ops.where(math_ops.not_equal(p_assignments, i))
                    flat_ids_not_in_i = array_ops.reshape(ids_not_in_i, [-1])
                    if sp_input.indices.dtype == dtypes.int64:
                        flat_ids_not_in_i = math_ops.cast(flat_ids_not_in_i, dtypes.int64)
                    else:
                        flat_ids_not_in_i = math_ops.cast(flat_ids_not_in_i, dtypes.int32)                    
                    s = array_ops.sparse_mask(sp_input, flat_ids_not_in_i)
                    if i < extras:
                        s._indices = math_ops.floor_mod(s.indices, ids_per_partition + 1)
                    else:
                        s._indices = math_ops.floor_mod(s.indices - extras, ids_per_partition)
                split_grads.append(s)
        return split_grads

    @staticmethod
    def _split_tensor_v2(value, num_splits, shape, axis=0, name=None):

        def _iter_slices(full_shape, num_slices, slice_dim):
            """Slices a given a shape along the specified dimension."""
            num_slices_with_excess = full_shape[slice_dim] % num_slices
            offset = [0] * len(full_shape)
            min_slice_len = full_shape[slice_dim] // num_slices
            for i in range(num_slices):
                shape = full_shape[:]
                shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
                yield offset[:], shape
                offset[slice_dim] += shape[slice_dim]

        split_grads = []
        for idx, (begin, var_shape) in enumerate(
                _iter_slices(shape.as_list(), num_splits, axis)):
            split_grads.append(array_ops.slice(value, begin, var_shape, name=name + f'_part_{idx}'))
        return split_grads

    @staticmethod
    def _prune_graphdef_node_inputs(node):
        """
        Remove empty inputs from a NodeDef.

        Args:
            node (NodeDef): a Graphdef node.
        """
        node.input[:] = [s for s in node.input if s]
